# 🧠 RAG-Docs — Retrieval-Augmented Generation sur documents

## 🎯 Objectif du projet
Ce projet implémente un système RAG (Retrieval‑Augmented Generation) permettant de poser des questions en langage naturel sur un corpus de documents (PDF, DOCX, TXT).  
Composants :
- Backend Flask : API pour indexation, upload et interrogation.
- Moteur d'embeddings + FAISS pour la recherche vectorielle.
- Google Gemini pour la génération de réponses.
- Frontend Streamlit pour l'interface utilisateur.

---

## ⚙️ Stack technique

### 🧩 Backend :
- **Python 3.10+**
- **Flask** – Framework web léger pour exposer les endpoints `/ask` et `/healthcheck`
- **LangChain** – Orchestration du pipeline RAG
- **FAISS** – Moteur de recherche vectorielle
- **SentenceTransformers** – Génération des embeddings
- **Google Generative AI (Gemini API)** – Modèle LLM pour la génération de texte
- **Pydantic** – Validation des modèles de requêtes/réponses

### 💻 Frontend :
- **Streamlit** – Interface utilisateur simple et rapide
- Champ de saisie pour poser des questions
- Affichage de la réponse et des sources documentaires

---

## 📁 Structure du projet
```
rag-docs/
├── backend/
│   ├── main.py           # API Flask principale
│   ├── rag_engine.py     # Extraction, nettoyage, embeddings, FAISS, retrieval
│   ├── models.py         # Schémas Pydantic
│   └── requirements.txt
├── frontend/
│   └── app.py            # Interface Streamlit
├── data/
│   └── raw_documents/    # Documents à indexer
├── .env.example
└── README.md
```

---

## 🧩 Installation (locale)

1. Cloner le dépôt
```bash
git clone <repo-url>
cd rag-docs
```

2. Créer et activer un virtualenv
- PowerShell (Windows)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
- cmd (Windows)
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```
- macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Installer les dépendances (backend + frontend)
```bash
pip install -r backend/requirements.txt
pip install streamlit requests
```

---

## 🔑 Configuration des clés API
Copier l'exemple de configuration et renseigner les clés nécessaires :
```bash
cp .env.example .env
```
`.env` :
```
GEMINI_API_KEY=Your_Key
API_BASE="http://localhost:8000"
# EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
DATA_DIR="../data/raw_documents"
FAISS_DIR="../data/vector_store"
EMBED_MODEL="all-MiniLM-L6-v2.gguf2.f16.gguf"
CHUNK_SIZE=500
CHUNK_OVERLAP=120
TOP_K=4
UPLOAD_FOLDER="../data/raw_documents"
```
- Ouvrir `.env` et remplacer `Your_Key` par votre clé Gemini personnelle.
  ⚠️ Important — Remplacez Your_Key par votre propre clé Gemini API obtenue sur Google AI Studio.  
Sans cette clé, le modèle de génération ne fonctionnera pas.
---

## 🚀 Démarrage

### Backend (développement)
```bash
cd backend
python main.py
```
- Serveur par défaut : http://localhost:8000
- Endpoint santé : `GET /healthcheck` → renvoie `{"status":"ok"}`

### Frontend (Streamlit)
```bash
cd frontend
streamlit run app.py
```
- Ouvrir l'URL affichée (ex : http://localhost:8501)

---

## 📡 Endpoints principaux

- GET /healthcheck  
  Vérifie l'état du service.

- POST /reindex  
  Reconstruit l'index FAISS depuis `data/raw_documents`.  
  - Mode synchrone (JSON final) : `POST /reindex?stream=false`  
  - Mode stream (SSE) : `POST /reindex` (consommer le flux)

- POST /upload  
  Permet d’uploader un fichier (PDF, DOCX, TXT) avec option de réindexation.  

- POST /ask  
  Poser une question au système RAG.

---

## 🧪 Démonstration d'usage (exemples)
- Login de démo : `admin` / `admin`

<img width="600"  alt="Screenshot 2025-10-17 150903" src="https://github.com/user-attachments/assets/714aeefa-b794-4522-8934-43552aa2c181" />

- Indexation complète via UI : `Indexation / Reindex` → `Reindexer tout le corpus`

<img width="400" alt="Screenshot 1" src="https://github.com/user-attachments/assets/c0dc0f34-4c14-44c2-bac1-453af5c8ee5f" /><br/>

<img width="400" alt="Screenshot 2" src="https://github.com/user-attachments/assets/e06d6fe9-bb22-499d-bc55-b34b3abf1e74" /><br/>

<img width="400" alt="Screenshot 3" src="https://github.com/user-attachments/assets/908c5b4a-8b65-468c-a7a3-c8ed0eb9eaa1" /><br/>

<img width="400" alt="Screenshot 4" src="https://github.com/user-attachments/assets/c700ea6c-2f4e-45a7-97c6-73dfe4d43610" /><br/>

<img width="600" alt="Screenshot 2025-10-17 151244" src="https://github.com/user-attachments/assets/8563ab24-2975-453f-a72c-1f2cb40a591c" /><br/>

<img width="550" alt="Screenshot 2025-10-17 153851" src="https://github.com/user-attachments/assets/729e2328-0d86-4c14-9bc5-5d8033cb2e9a" />

- Upload + forcer index : `Uploader un document` → cocher `Forcer réindexation après upload` → `Upload et Reindex`

<img width="550" alt="Screenshot 2025-10-17 153951" src="https://github.com/user-attachments/assets/7657c59d-6c60-4e57-8d85-4f22414338fe" />

- Poser une question : remplir le champ puis `Poser la question`

<img width="400" alt="Screenshot 2025-10-17 163057" src="https://github.com/user-attachments/assets/d13dda11-dd64-4b15-8a9a-189b2c667484" />

- Historique de conversation :

<img width="400" alt="Screenshot 2025-10-17 163657" src="https://github.com/user-attachments/assets/649182f9-4c21-47fb-9bc5-cf75baae4f83" />

---

## ⚠️ Notes
- L'indexation peut prendre plusieurs minutes selon le volume et le CPU.  
---

## ✅ Fonctionnalités implémentées
- Extraction texte PDF / DOCX / TXT
- Nettoyage et normalisation texte
- Chunking sémantique
- Embeddings via sentence-transformers (ou wrapper LangChain)
- Indexation FAISS locale (persist)
- Endpoints REST pour ask/upload/reindex
- Frontend Streamlit avec upload, reindex, question/answer UI
- Option Gemini LLM si clé fournie (fallback extractif sinon)
  
---

## 👩‍💻 Auteur
Khadija ZOUHAIR  
Étudiante ingénieure en Informatique & Data Science — ENSA Khouribga

---

## 🔚 Conclusion
Ce dépôt fournit un pipeline RAG complet pour interroger votre propre corpus documentaires.
