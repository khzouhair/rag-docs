import streamlit as st
import requests
import os

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="RAG Docs QA", layout="centered")

# --- session init ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []  

# simple local login (demo: Username: admin / Password: admin)
def _do_login():
    user = st.session_state.get("login_user", "")
    pwd = st.session_state.get("login_pwd", "")
    if user == "admin" and pwd == "admin":
        st.session_state["logged_in"] = True
        st.session_state["user"] = user
        st.success("Logged in")
    else:
        st.error("Invalid credentials")

def _do_logout():
    st.session_state["logged_in"] = False
    st.session_state["user"] = None
    st.button("Se connecter")
    st.success("Déconnecté")

# Login form
if not st.session_state["logged_in"]:
    st.title("RAG Docs QA — Connexion")
    with st.form("login_form"):
        st.text_input("Nom d'utilisateur", key="login_user")
        st.text_input("Mot de passe", key="login_pwd", type="password")
        submitted = st.form_submit_button("Se connecter")
        if submitted:
            _do_login()
    st.markdown("Nom d'utilisateur: **admin** / Mot de passe: **admin**")
    if not st.session_state["logged_in"]:
        st.stop()

# --- main UI ---
st.header("📚 RAG Documentaire — Q/A")
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Se déconnecter"):
        _do_logout()
        st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.stop()

with st.expander("Indexation / Reindex"):
    if st.button("Reindexer tout le corpus (rebuild vector store)"):
        try:
            with st.spinner("Indexation en cours (peut prendre plusieurs minutes)..."):
                r = requests.post(f"{API_BASE}/reindex?stream=false", timeout=900)
            if r.ok:
                try:
                    j = r.json()
                    st.success(f"Indexation terminée — chunks: {j.get('chunks_indexed')}")
                except Exception:
                    st.success("Indexation terminée.")
            else:
                try:
                    detail = r.json()
                except Exception:
                    detail = r.text
                st.error(f"Erreur reindex: {detail}")
        except Exception as e:
            st.error(f"Erreur reindex: {e}")


st.markdown("---")

question = st.text_area("Ta question :", height=140, placeholder="Entrez une question sur les documents indexés ici...")
top_k = st.number_input("Nombre de sources (top_k)", min_value=1, max_value=50, value=4, step=1)
allow_pickle = st.checkbox("Autoriser la désérialisation du store local (pickle) — attention sécurité", value=False)


import concurrent.futures
import time

def _post_request(url, payload, timeout):
    """Synchronous request wrapper for executor thread."""
    return requests.post(url, json=payload, timeout=timeout)

if st.button("Poser la question"):
    q = question.strip()
    if not q:
        st.warning("Entrez une question.")
        st.stop()

    # prepare history payload
    recent = st.session_state.history[-10:]
    hist_msgs = []
    for item in recent:
        u_text = str(item.get("question", "") or "")
        a_text = str(item.get("answer", "") or "")
        a_sources = item.get("sources", []) or []
        normalized_sources = []
        for s in a_sources:
            try:
                normalized_sources.append({
                    "source": str(s.get("source", "history")),
                    "text": str(s.get("text", ""))
                })
            except Exception:
                continue
        hist_msgs.append({"role": "user", "text": u_text})
        hist_msgs.append({"role": "assistant", "text": a_text, **({"sources": normalized_sources} if normalized_sources else {})})

    payload = {
        "question": q,
        "top_k": int(top_k),
        "allow_dangerous_deserialization": bool(allow_pickle),
        "history": hist_msgs
    }

    timeout_seconds = 130
    start_t = time.time()

    # submit request in background thread using ThreadPoolExecutor and poll future
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_post_request, f"{API_BASE}/ask", payload, 120)

        progress = st.progress(0)
        status = st.empty()
        pct = 0

        # poll loop: animate progress while waiting for result
        while True:
            if future.done():
                break
            elapsed = time.time() - start_t
            if elapsed >= timeout_seconds:
                # Timeout: try cancel and break
                try:
                    future.cancel()
                except Exception:
                    pass
                break
            # simple indeterminate animation up to 90%
            pct = (pct + 7) % 91
            progress.progress(pct)
            status.text(f"Recherche en cours... {int(elapsed)}s")
            time.sleep(0.12)

        # ensure UI shows completion
        progress.progress(100)
        status.text("Recherche terminée, récupération des résultats...")
        time.sleep(0.12)
        progress.empty()
        status.empty()

        # get result / handle errors
        resp = None
        try:
            if future.done() and not future.cancelled():
                resp = future.result()
            else:
                st.error("La requête a expiré (timeout). Vérifie le backend ou augmente le timeout.")
                st.stop()
        except Exception as exc:
            st.error(f"Erreur pendant la requête : {exc}")
            st.stop()

    # no response object
    if resp is None:
        st.error("Aucune réponse reçue du serveur. Vérifie que le backend est démarré et accessible.")
        st.stop()

    # HTTP error handling
    if not getattr(resp, "ok", False):
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text or f"status_code={getattr(resp,'status_code','??')}"
        st.error(f"Erreur API: {getattr(resp,'status_code','??')} — {detail}")
        st.stop()

    # parse JSON
    try:
        data = resp.json()
    except Exception:
        st.error("Réponse invalide du serveur (non-JSON).")
        st.stop()

    # existing response processing
    answer_raw = data.get("answer", "") or ""
    sources = data.get("sources", []) or []

    try:
        import re
        m = re.search(r'(?i)based on recent conversation\s*:\s*', answer_raw)
        if m:
            start = m.end()
            m2 = re.search(r'(?i)based on recent conversation\s*:\s*', answer_raw[start:])
            end = start + m2.start() if m2 else len(answer_raw)
            block = answer_raw[start:end].strip()
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            assistant_parts = []
            for ln in lines:
                low = ln.lower()
                if low.startswith("assistant:"):
                    assistant_parts.append(ln[len("assistant:"):].strip())
                elif low.startswith("user:"):
                    continue
                else:
                    if assistant_parts:
                        assistant_parts[-1] += " " + ln
                    else:
                        assistant_parts.append(ln)
            assistant_text = " ".join(p for p in assistant_parts).strip()
            display_answer = "Based on recent conversation:\n\n" + assistant_text if assistant_text else "Based on recent conversation:"
        else:
            cleaned_lines = []
            for ln in answer_raw.splitlines():
                if ln.strip().lower().startswith("user:"):
                    continue
                cleaned_lines.append(ln)
            display_answer = "\n".join(cleaned_lines).strip()
    except Exception:
        display_answer = answer_raw

    # filter sources
    file_sources = []
    seen = set()
    for s in sources:
        try:
            src = str(s.get("source", "") or "")
            txt = str(s.get("text", "") or "")
        except Exception:
            continue
        if not src or src.lower() == "history":
            continue
        if any(src.lower().endswith(ext) for ext in [".pdf", ".docx", ".txt"]) or ("/" in src) or ("\\" in src):
            key = (src, txt[:200])
            if key not in seen:
                seen.add(key)
                file_sources.append({"source": src, "text": txt})
    if not file_sources:
        for s in sources:
            src = s.get("source", "") or ""
            txt = s.get("text", "") or ""
            if str(src).lower() != "history":
                key = (str(src), str(txt)[:200])
                if key not in seen:
                    seen.add(key)
                    file_sources.append({"source": str(src), "text": str(txt)})

    # store in history 
    last = st.session_state.history[-1] if st.session_state.history else None
    if not (last and last.get("question") == q and last.get("answer") == display_answer):
        st.session_state.history.append({
            "question": q,
            "answer": display_answer,
            "sources": file_sources
        })
        st.session_state.history = st.session_state.history[-200:]

    # display results
    st.subheader("Réponse")
    st.markdown(display_answer)

    prompt = data.get("prompt", "")
    if prompt:
        with st.expander("Prompt utilisé (debug)"):
            st.code(prompt)

    st.subheader(f"Sources ({len(sources)})")
    for i, s in enumerate(sources):
        src = s.get("source", "unknown")
        txt = s.get("text", "") or ""
        with st.expander(f"[{i+1}] {src}"):
            st.write(txt)
        
st.markdown("---")
st.header("Uploader un document")
st.markdown("Formats acceptés: .pdf, .docx, .txt. Option: forcer la réindexation après upload (peut prendre plusieurs minutes).")
uploaded_file = st.file_uploader("Choisir un fichier", type=["pdf", "docx", "txt"])
force_reindex = st.checkbox("Forcer réindexation après upload (force=true)", value=False)
if uploaded_file is not None:
    st.write(f"Fichier sélectionné: {uploaded_file.name} ({uploaded_file.type})")
    if st.button("Upload et (optionnel) Reindex"):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
            params = {"force": "true"} if force_reindex else {}
            with st.spinner("Upload en cours..."):
                r = requests.post(f"{API_BASE}/upload", params=params, files=files, timeout=600)
            if r.ok:
                j = r.json()
                if j.get("status") == "ok":
                    st.success(f"Fichier uploadé: {j.get('file_saved')} — chunks indexed: {j.get('chunks_indexed', 'skipped')}")
                else:
                    st.info(str(j))
            else:
                st.error(f"Erreur upload: {r.status_code} — {r.text}")
        except Exception as e:
            st.error(f"Erreur pendant l'upload: {e}")
st.markdown("---")

st.header("Historique")
if not st.session_state.history:
    st.write("Aucun historique.")
else:
    for i, item in enumerate(reversed(st.session_state.history[-50:])):
        st.markdown(f"**Q:** {item.get('question')}")
        st.markdown(f"**A:** {item.get('answer')}")
        st.markdown("---")
st.caption("Frontend minimal Streamlit pour démonstration RAG Docs QA.")
