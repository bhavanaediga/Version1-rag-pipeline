import asyncio

import httpx
import chainlit as cl

API_BASE = "http://localhost:8000"


async def _fetch_ready_docs() -> list[dict]:
    """Return all documents with processing_status == done."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.get(f"{API_BASE}/documents")
            docs = res.json()
            return [d for d in docs if d.get("processing_status") == "done"]
    except Exception:
        return []


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("active_doc_ids", [])

    ready_docs = await _fetch_ready_docs()

    if not ready_docs:
        await cl.Message(
            content=(
                "Welcome to **PlanSmartAI**.\n\n"
                "No documents in the database yet. "
                "Upload a construction PDF using the paperclip icon to get started."
            )
        ).send()
        return

    # Build numbered list of existing documents
    lines = ["Welcome to **PlanSmartAI**.\n"]
    lines.append("The following documents are already indexed and ready to query:\n")
    for i, doc in enumerate(ready_docs, 1):
        name = doc["file_name"]
        ftype = doc["file_type"]
        doc_id = doc["id"]
        lines.append(f"**{i}.** {name} `({ftype})` — `{doc_id[:8]}...`")

    lines.append(
        "\nType the **number(s)** of the document(s) you want to query "
        "(e.g. `1` or `1,2`), or type `new` to upload a new PDF."
    )

    cl.user_session.set("ready_docs", ready_docs)
    cl.user_session.set("awaiting_selection", True)

    await cl.Message(content="\n".join(lines)).send()


@cl.on_message
async def on_message(message: cl.Message):
    files = [e for e in (message.elements or []) if isinstance(e, cl.File)]

    # ── File upload attached ───────────────────────────────────────────────────
    if files:
        await _handle_upload(message, files)
        return

    # ── Awaiting document selection from the startup list ─────────────────────
    if cl.user_session.get("awaiting_selection"):
        await _handle_selection(message)
        return

    # ── Normal query ──────────────────────────────────────────────────────────
    await _handle_query(message)


async def _handle_selection(message: cl.Message):
    text = message.content.strip().lower()

    if text == "new":
        cl.user_session.set("awaiting_selection", False)
        await cl.Message(
            content="Sure — upload a PDF using the paperclip icon and I'll process it."
        ).send()
        return

    ready_docs = cl.user_session.get("ready_docs") or []
    selected_ids = []
    selected_names = []

    try:
        indices = [int(x.strip()) for x in text.split(",")]
        for idx in indices:
            if 1 <= idx <= len(ready_docs):
                doc = ready_docs[idx - 1]
                selected_ids.append(doc["id"])
                selected_names.append(doc["file_name"])
            else:
                await cl.Message(
                    content=f"No document number {idx}. Please pick from the list above."
                ).send()
                return
    except ValueError:
        await cl.Message(
            content="Please type a number (or numbers separated by commas) from the list, or `new`."
        ).send()
        return

    cl.user_session.set("active_doc_ids", selected_ids)
    cl.user_session.set("awaiting_selection", False)

    names_str = ", ".join(f"**{n}**" for n in selected_names)
    await cl.Message(
        content=f"Loaded {names_str}. Ask me anything about it."
    ).send()


async def _handle_upload(message: cl.Message, files: list):
    doc_ids = cl.user_session.get("active_doc_ids") or []
    cl.user_session.set("awaiting_selection", False)

    for f in files:
        res = await cl.AskUserMessage(
            content=(
                f"Got **{f.name}**. "
                "Is this a **blueprint** (visual drawing/floor plan) or a **spec** (text document)? "
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

        await cl.Message(
            content=f"Got it! Processing **{f.name}** — I'll let you know when it's ready."
        ).send()

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
        cl.user_session.set("active_doc_ids", doc_ids)

        await _poll_status(document_id)


async def _poll_status(document_id: str):
    max_attempts = 120  # up to 10 minutes, polling every 5s
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
            await cl.Message(
                content="Something went wrong processing that file. Please try again."
            ).send()
            return

    await cl.Message(
        content="Processing is taking longer than expected. Check back soon."
    ).send()


async def _handle_query(message: cl.Message):
    doc_ids = cl.user_session.get("active_doc_ids") or []

    if not doc_ids:
        await cl.Message(
            content="No document selected. Type a number from the list above, or upload a new PDF."
        ).send()
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
        page_or_chunk = src.get("page_or_chunk", "")
        source_lines.append(f"- {label} (ref: {page_or_chunk})")

    sources_block = (
        "\n\n**Sources:**\n" + "\n".join(source_lines) if source_lines else ""
    )
    confidence_note = f"\n\n*Confidence: {confidence:.2f}*"

    thinking_msg.content = answer + sources_block + confidence_note
    await thinking_msg.update()
