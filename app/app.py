import streamlit as st
import os
from sentence_store.main import Embedder
from sentence_store.tools import remove_dir

UPLOAD_DIR = './UPLOAD_DIR/'


def handle_uploaded():
    if 'uploaded_file' not in st.session_state:
        return None
    if st.session_state.uploaded_file is None:
        return None
    fpath = save_uploaded_file()
    assert os.path.exists(fpath), fpath
    print('FPATH:', fpath)

    suf = fpath[-4:].lower()
    # fname = fpath[:-4]
    if suf == ".pdf" or suf == ".txt":
        return fpath
    else:
        st.write("UPLOAD .txt / .pdf file!")


def save_uploaded_file():
    upload_dir = UPLOAD_DIR
    fname = st.session_state.uploaded_file.name
    fpath = os.path.join(upload_dir, fname)
    if os.path.exists(fpath):
        return fpath
    os.makedirs(upload_dir, exist_ok=True)
    with open(fpath, "wb") as f:
        f.write(st.session_state.uploaded_file.getbuffer())
    return fpath


with st.sidebar:
    st.write('**Sentence Store Test App**')
    doc_type = st.radio('Document type?', ('local pdf or txt file', 'url'), index=0, horizontal=True)

    if doc_type == 'local pdf or txt file':
        st.session_state.uploaded_file = st.file_uploader(
            "SELECT FILE", type=["txt", "pdf"]
        )
        doc_name = handle_uploaded()
        # st.write('DOC:', doc_name)
        print('DOC:', doc_type, doc_name)

        if doc_name is None:
            pass
        elif doc_name.lower().endswith('.pdf'):
            doc_type = 'pdf'
        elif doc_name.lower().endswith('.txt'):
            doc_type = 'txt'
        else:
            st.write('Unable to process:', doc_name)
            doc_name = None

    else:
        doc_name = st.text_input('Link to document name?', value="")

    store_it = st.sidebar.button('Store document!')

    answer_count = st.slider('Number of answers?',
                             min_value=2, max_value=20, value=4)

    quest = st.text_area(label='Query', value="", key='quest')

query_it = st.sidebar.button('Query document!')
clear_it = st.sidebar.button('Clear caches!')

if doc_name:
    embedder = Embedder(cache_name=doc_name)


def process_it():
    if not doc_name: return

    if store_it:
        embedder.store_doc(doc_type, doc_name)
        st.write('TIMES:', embedder.times)
    elif clear_it:
        remove_dir(embedder.CACHES)
        remove_dir(UPLOAD_DIR)
    elif query_it:
        if quest:
            embedder.store_doc(doc_type, doc_name)
            answers = embedder.query(quest, top_k=answer_count)
            st.write('MATCHING SENTENCES:')
            st.write(answers)
            st.write('TIMES:', embedder.times)
        else:
            st.write('Please enter a query!')
    else:
        st.write('Ready!')


if __name__ == "__main__":
    process_it()
