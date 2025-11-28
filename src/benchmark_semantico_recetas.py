import streamlit as st
from Retrieval import retrieve
from VectorStore import vector_store_config
from qdrant_client import QdrantClient

# ============================
# Inicializar Qdrant
# ============================
client = QdrantClient(path="../data/tmp/langchain_qdrant")

st.set_page_config(page_title="RAG Biotechnologies", layout="centered")
st.title("🔬 RAG Biotechnologies")
st.write("Realiza preguntas basadas en los archivos indexados.")

# ============================
# Cargar configuración del vector store
# ============================
vector_store_config(client)

# ============================
# Input del usuario
# ============================
query = st.text_input("Pregunta sobre los archivos", placeholder="Ej: ¿Qué es un anticuerpo monoclonal?")

# Botón para ejecutar consulta
if st.button("Enviar"):

    if not query.strip():
        st.warning("⚠️ Debes escribir una pregunta antes de enviar.")
    else:
        st.write("🔍 Buscando respuesta...")

        try:
            response = retrieve(client, query)
            st.success("Respuesta encontrada:")
            st.write(response)
        except Exception as e:
            st.error("❌ Error al ejecutar el RAG")
            st.error(str(e))

# ============================
# Instrucción para ejecutar
# ============================
st.markdown("---")
st.caption("Ejecuta con:  `streamlit run main.py`")


