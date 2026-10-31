"""Optional Streamlit client for the Know-It API."""

from __future__ import annotations

import os
from uuid import UUID

import httpx
import streamlit as st

from app.ui_client import KnowItApiClient


def main() -> None:
    st.set_page_config(page_title="Know-It", page_icon="📚")
    st.title("Know-It")
    st.caption("Ask questions about the indexed technical documentation.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    if st.sidebar.button("Start a new conversation"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_sources(message.get("sources", []))
                if not message.get("feedback_sent"):
                    vote = st.feedback("thumbs", key=f"feedback-{index}")
                    if vote is not None:
                        rating = "up" if vote == 1 else "down"
                        try:
                            _client().feedback(UUID(message["answer_id"]), rating)
                        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                            st.error(str(exc))
                        else:
                            message["feedback_sent"] = True
                            st.rerun()

    if question := st.chat_input("Ask about the documentation"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                response = _client().query(question, st.session_state.session_id)
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.session_state.session_id = UUID(response["session_id"])
                st.markdown(response["answer"])
                _render_sources(response.get("sources", []))
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", []),
                        "answer_id": response["answer_id"],
                        "feedback_sent": False,
                    }
                )


@st.cache_resource
def _client() -> KnowItApiClient:
    return KnowItApiClient(os.getenv("KNOW_IT_API_URL", "http://127.0.0.1:8000"))


def _render_sources(sources: list[dict[str, object]]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for source in sources:
            label = source.get("title") or source.get("source")
            origin = source.get("origin", "corpus")
            st.markdown(f"- [{label}]({source['source']}) · `{origin}`")


if __name__ == "__main__":
    main()
