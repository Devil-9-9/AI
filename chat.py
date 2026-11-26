import streamlit as st
import torch

st.title("💎 Diamond Brain - Chat Karo")

# 1. Dimag load karo (training wali file)
@st.cache_resource
def load_model():
    from model.brain import DiamondTransformerBrain
    from model.tokenizer import DiamondTokenizer
    model = DiamondTransformerBrain()
    try:
        # Agar training ho gayi to ye file milegi
        model.load_state_dict(torch.load("diamond_brain_500.pt", map_location="cpu"))
        st.success("Trained Dimag Load Ho Gaya!")
    except:
        st.warning("Abhi Training nahi hui, naya khali dimag use ho raha hai")
    model.eval()
    return model, DiamondTokenizer()

model, tok = load_model()

# 2. Chat History yaad rakho
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 3. Chat karne ki jagah
if prompt := st.chat_input("Diamond se kuch pucho..."):
    # Tumhara message dikhao
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # AI ka jawab banao
    with st.chat_message("assistant"):
        with st.spinner("Diamond soch raha hai..."):
            ids = torch.tensor([tok.encode(prompt)])
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=30)
            reply = tok.decode(out[0].tolist())
            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
