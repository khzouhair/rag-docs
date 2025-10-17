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

<img width="600"  alt="Screenshot 2025-10-17 150903" src="https://github.com/user-attachments/assets/4c8a81da-72db-4e94-99b0-92517ae4aa18" />


- Indexation complète via UI : `Indexation / Reindex` → `Reindexer tout le corpus`

<img width="400"  alt="Screenshot 1" src="https://github.com/user-attachments/assets/2c5b05f5-ae5f-4cb1-a809-30a29f46ea47" /><br/>

<img width="400"  alt="Screenshot 2" src="https://github.com/user-attachments/assets/5c398701-8699-42c3-bbe1-5258510bd59f" /><br/>

<img width="400"  alt="Screenshot 3" src="https://github.com/user-attachments/assets/95473357-a495-4e44-8271-27e66beab354" /><br/>

<img width="400"  alt="Screenshot 4" src="https://github.com/user-attachments/assets/f85fc155-846d-4d09-a7aa-9a62d614fd64" /><br/>

<img width="400"  alt="Screenshot 2025-10-17 151244" src="https://github.com/user-attachments/assets/42956907-b861-40f4-8a90-7524367487ca" /><br/>

<img width="400" alt="Screenshot 2025-10-17 153851" src="https://github.com/user-attachments/assets/6c298414-e6ce-473c-9a42-6c11d9732be4" />


- Upload + forcer index : `Uploader un document` → cocher `Forcer réindexation après upload` → `Upload et Reindex`

<img width="500"  alt="Screenshot 5" src="https://github.com/user-attachments/assets/1dea14fe-5250-49ee-817b-4cb748c8cc59" /><br/>

<img width="500"  alt="Screenshot 6" src="https://github.com/user-attachments/assets/06c12df4-c34b-4a3a-ab6f-aa3dcdf09205" />


- Poser une question : remplir le champ puis `Poser la question`

<img width="400" alt="Screenshot 2025-10-17 163057" src="https://github.com/user-attachments/assets/85333dca-8062-49e3-9bde-fae1fea62f38" />


- Historique de conversation :

<img width="578" height="644" alt="Screenshot 2025-10-17 163657" src="https://github.com/user-attachments/assets/19d1f656-fcdc-426e-84eb-d60d3294aef2" />


---


## ⚠️ Notes
- L'indexation peut prendre plusieurs minutes selon le volume et le CPU.  
- Vous devez uploader les fichiers que vous souhaitez interroger via le système.  
- Si vous rencontrez un problème lors de l'exécution, veuillez me contacter : khadijazouhair2004@gmail.com

  
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
