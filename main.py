import gradio as gr
from chatllm.config import env
from chatllm.ui import build_demo


def main() -> None:
    demo = build_demo()
    demo.launch(
        server_name=env("HOST", "0.0.0.0"),
        server_port=int(env("PORT", "7860")),
        share=env("SHARE", "0") == "1",
        footer_links=["gradio"],
        theme=gr.themes.Soft(),
        css="""
        footer {display: none !important;}
        #api-button {display: none !important;}
        """,
    )


if __name__ == "__main__":
    main()
