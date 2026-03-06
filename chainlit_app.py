import asyncio

import httpx
import chainlit as cl

API_BASE = "http://localhost:8000"


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("uploaded_document_ids", [])
    await cl.Message(
        content=(
            "Welcome to **PlanSmartAI**. "
            "Upload a construction PDF using the paperclip icon, then ask me anything about it."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    # Chainlit 2.x: file attachments are in message.elements as cl.File objects
    files = [e for e in (message.elements or []) if isinstance(e, cl.File)]

    if files:
        await _handle_upload(message, files)
    else:
        await _handle_query(message)


async def _handle_upload(message: cl.Message, files: list):
    doc_ids = cl.user_session.get("uploaded_document_ids") or []

    for f in files:
        res = await cl.AskUserMessage(
            content=(
                f"Got **{f.name}**. "
                "Is this a **blueprint** (visual drawing/floor plan) or a **spec** (text document/code)? "
                "Reply with `blueprint` or `spec`."
            ),
            timeout=60,
        ).send()

        if not res:
            await cl.Message(content="No response received. Upload cancelled.").send()
            continue

        file_type = res["output"].strip().lower()
        if file_type not in {"blueprint", "spec"}:
            await cl.Message(content="Please reply with exactly `blueprint` or `spec`.").send()
            continue

        await cl.Message(content="Got it! Processing your document... I'll let you know when it's ready.").send()

        async with httpx.AsyncClient(timeout=60) as client:
            with open(f.path, "rb") as fh:
                response = await client.post(
                    f"{API_BASE}/upload",
                    files={"file": (f.name, fh, "application/pdf")},
                    data={"file_type": file_type},
                )

        if response.status_code != 200:
            await cl.Message(content=f"Upload failed: {response.text}").send()
            continue

        data = response.json()
        document_id = data["document_id"]
        doc_ids.append(document_id)
        cl.user_session.set("uploaded_document_ids", doc_ids)

        await _poll_status(document_id)


async def _poll_status(document_id: str):
    max_attempts = 120  # up to 10 minutes
    last_detail = ""
    for _ in range(max_attempts):
        await asyncio.sleep(5)
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                res = await client.get(f"{API_BASE}/documents/{document_id}/status")
                data = res.json()
                status = data.get("status", "")
                detail = data.get("progress_detail", "")
            except Exception:
                status = "unknown"
                detail = ""

        if detail and detail != last_detail:
            await cl.Message(content=f"⏳ {detail}").send()
            last_detail = detail

        if status == "done":
            await cl.Message(content="Document ready! Ask me anything about it.").send()
            return
        if status == "failed":
            await cl.Message(content="Something went wrong processing that file. Please try again.").send()
            return

    await cl.Message(content="Processing is taking longer than expected. Check back soon.").send()


async def _handle_query(message: cl.Message):
    doc_ids = cl.user_session.get("uploaded_document_ids") or []

    if not doc_ids:
        await cl.Message(content="Please upload a document first.").send()
        return

    thinking_msg = cl.Message(content="Thinking...")
    await thinking_msg.send()

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            response = await client.post(
                f"{API_BASE}/query",
                json={"query": message.content, "document_ids": doc_ids},
            )
            data = response.json()
        except Exception as exc:
            thinking_msg.content = f"Error contacting backend: {exc}"
            await thinking_msg.update()
            return

    answer = data.get("answer", "No answer returned.")
    cited_sources = data.get("cited_sources", [])
    confidence = data.get("confidence_score", 0.0)

    source_lines = []
    for src in cited_sources:
        label = src.get("label", "")
        doc_id = src.get("document_id", "")
        page_or_chunk = src.get("page_or_chunk", "")
        source_lines.append(f"- {label} (doc: `{doc_id[:8]}...`, ref: {page_or_chunk})")

    sources_block = (
        "\n\n**Sources:**\n" + "\n".join(source_lines) if source_lines else ""
    )
    confidence_note = f"\n\n*Confidence: {confidence:.2f}*"

    thinking_msg.content = answer + sources_block + confidence_note
    await thinking_msg.update()
