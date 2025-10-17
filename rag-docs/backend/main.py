import os
import logging
from flask import Flask, request, jsonify
from pydantic import ValidationError
from dotenv import load_dotenv
from models import AskRequest, AskResponse, SourceOut
from rag_engine import full_preprocess_and_index, answer_question_flow, DATA_DIR
from werkzeug.utils import secure_filename

FAISS_DIR = os.getenv("FAISS_DIR", "../data/vector_store")


load_dotenv()
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# Upload config and allowed extensions
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "../data/raw_documents")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXT = {".pdf", ".docx", ".txt"}

def _vector_store_exists(path: str) -> bool:
    try:
        if not path or not os.path.exists(path):
            return False
        for n in os.listdir(path):
            if n.startswith("index") or n.endswith(".faiss") or n.endswith(".pkl"):
                return True
        return False
    except Exception:
        return False

@app.get("/healthcheck")
def health():
    return jsonify({"status": "ok"})

@app.post("/reindex")
def reindex():
    force = request.args.get("force", "false").lower() == "true"
    if _vector_store_exists(FAISS_DIR) and not force:
        return jsonify({"status": "skipped", "detail": "vector store exists; use ?force=true to rebuild"}), 200
    try:
        count = full_preprocess_and_index(DATA_DIR, FAISS_DIR)
        return jsonify({"status": "ok", "chunks_indexed": count}), 200
    except Exception as e:
        logging.exception("Reindex failed")
        return jsonify({"status": "error", "detail": str(e)}), 500


@app.post("/ask")
def ask():
    import json
    try:
        body = request.get_json(force=True) or {}
    except Exception as e:
        logging.exception("Invalid JSON in /ask")
        return jsonify({"error": "invalid json", "detail": str(e)}), 400

    # sanitize history if present 
    raw_hist = body.get("history")
    if isinstance(raw_hist, list):
        sanitized = []
        for h in raw_hist:
            try:
                role = str(h.get("role", "user"))
                t = h.get("text", "")
                text = json.dumps(t, ensure_ascii=False) if isinstance(t, (dict, list)) else str(t)
                srcs_in = h.get("sources") or []
                sanitized_srcs = []
                if isinstance(srcs_in, list):
                    for s in srcs_in:
                        try:
                            src_name = str(s.get("source", "history"))
                            src_text_raw = s.get("text", "")
                            src_text = json.dumps(src_text_raw, ensure_ascii=False) if isinstance(src_text_raw, (dict, list)) else str(src_text_raw)
                            sanitized_srcs.append({"source": src_name, "text": src_text})
                        except Exception:
                            continue
                entry = {"role": role, "text": text}
                if sanitized_srcs:
                    entry["sources"] = sanitized_srcs
                sanitized.append(entry)
            except Exception:
                continue
        body["history"] = sanitized

    # validate payload
    try:
        payload = AskRequest(**body)
    except ValidationError as e:
        logging.exception("Validation error in /ask")
        return jsonify({"error": e.errors()}), 400

    chat_history = body.get("history")
    top_k = int(payload.top_k) if getattr(payload, "top_k", None) is not None else int(os.getenv("TOP_K", "4"))

    allow_flag = bool(payload.allow_dangerous_deserialization)
    if not allow_flag:
        allow_flag = os.getenv("TRUST_LOCAL_VECTOR_STORE", "false").lower() == "true"

    logging.info("ask: question=%s top_k=%s allow_dangerous_deserialization=%s", payload.question[:120], top_k, allow_flag)

    try:
        res = answer_question_flow(
            payload.question,
            faiss_dir=FAISS_DIR,
            top_k=top_k,
            allow_dangerous_deserialization=allow_flag,
            chat_history=chat_history,
        )

        if not isinstance(res, dict):
            logging.error("answer_question_flow returned non-dict: %r", res)
            return jsonify({"error": "internal", "detail": "invalid response from rag_engine"}), 500

        sources = res.get("sources") or []
        # ensure sources are simple dicts
        safe_sources = []
        for s in sources:
            try:
                safe_sources.append({"source": str(s.get("source", "unknown")), "text": str(s.get("text", ""))})
            except Exception:
                continue

        resp_model = AskResponse(
            answer=res.get("answer", "") or "",
            sources=[SourceOut(**s) for s in safe_sources],
            prompt=res.get("prompt"),
        )
        return jsonify(resp_model.dict()), 200

    except FileNotFoundError as e:
        logging.exception("Vector store not found")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logging.exception("Unhandled error in /ask")
        return jsonify({"error": "internal", "detail": str(e)}), 500


@app.post("/upload")
def upload():
    if "file" not in request.files:
        return {"error": "no file part"}, 400
    f = request.files["file"]
    if f.filename == "":
        return {"error": "no selected file"}, 400
    filename = secure_filename(f.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return {"error": f"extension {ext} not allowed"}, 400
    dest = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(dest)
    
    try:
        count = full_preprocess_and_index(DATA_DIR, FAISS_DIR)
        return {"status": "ok", "file_saved": filename, "chunks_indexed": count}
    except Exception as e:
        return {"status": "error", "detail": str(e)}, 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)