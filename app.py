import streamlit as st
from openai import OpenAI
import os, json
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from dotenv import load_dotenv
from chromadb.config import Settings # 新增导入

load_dotenv()

# --- 1. 初始化配置 ---
client = OpenAI(api_key=os.getenv("ZHIPU_API_KEY"), base_url=os.getenv("ZHIPU_BASE_URL"))

# 初始化向量库
# 修改后的初始化方式（增加了 settings 确保兼容性）
chroma_client = chromadb.PersistentClient(
    path="./my_vectordb",
    settings=Settings(allow_reset=True, anonymized_telemetry=False)
)

# 重新获取 collection
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(name="pdf_docs", embedding_function=default_ef)

# --- 2. 页面设置 ---
st.set_page_config(page_title="专业 RAG 助手", page_icon="🧠")
st.title("🧠 专业级 RAG 知识库助手")

# --- 3. 侧边栏：文件处理 ---
with st.sidebar:
    st.header("知识库管理")
    uploaded_file = st.file_uploader("上传 PDF 文档", type="pdf")
    
    if uploaded_file:
        if st.button("开始学习文档"):
            with st.spinner("正在解析并建立索引..."):
                # 解析 PDF
                reader = PdfReader(uploaded_file)
                all_chunks = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        # 简单切片
                        chunks = [text[i:i+500] for i in range(0, len(text), 400)]
                        all_chunks.extend(chunks)
                
                # 存入向量库
                ids = [f"{uploaded_file.name}_{i}" for i in range(len(all_chunks))]
                collection.add(documents=all_chunks, ids=ids)
                st.success(f"学习完毕！共存入 {len(all_chunks)} 个知识点。")

    if st.button("清空知识库"):
        chroma_client.delete_collection("pdf_docs")
        collection = chroma_client.get_or_create_collection(name="pdf_docs", embedding_function=default_ef)
        st.session_state.messages = []
        st.rerun()

# --- 4. 聊天逻辑 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("向知识库提问..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- 关键步骤：检索 ---
    with st.spinner("正在检索相关知识..."):
        results = collection.query(query_texts=[prompt], n_results=3)
        context = "\n".join(results['documents'][0])
    
    # --- 关键步骤：构造 Prompt ---
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        # 即使历史很长，AI 也只会被喂入最相关的 context
        messages = [
            {"role": "system", "content": f"你是一个基于文档回答问题的专家。以下是检索到的相关内容：\n\n{context}\n\n请结合以上内容回答，如果内容无关，请告知用户。"},
            {"role": "user", "content": prompt}
        ]
        
        response = client.chat.completions.create(
            model="glm-4",
            messages=messages,
            stream=True
        )
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
                placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # 定义一个模拟的工具函数
def get_weather(city: str):
    # 实际开发中这里会调用天气 API
    if "北京" in city:
        return "北京晴，25°C，微风。"
    return f"{city}的天气暂时查不到，建议带伞。"

# 修改 AI 调用部分（简化版逻辑）
# 告诉 AI 它有哪些工具可以用
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如北京"}
                }
            }
        }
    }
]

# 当 AI 返回 tool_calls 时，你在后台执行函数并把结果回传给 AI