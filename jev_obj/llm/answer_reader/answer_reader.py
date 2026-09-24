import ast


def _parse_cached_answer(raw, label):
    answer = ast.literal_eval(raw)
    if not isinstance(answer, list) or len(answer) < 2:
        raise ValueError(f"Invalid cached answer for {label}")
    *related, threshold, room = answer
    if not all(isinstance(item, str) for item in related):
        raise ValueError(f"Invalid related objects for {label}")
    if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid confidence threshold for {label}")
    if not isinstance(room, str):
        raise ValueError(f"Invalid room prior for {label}")
    return related, room, float(threshold)


def read_answer(llm_answer_path, llm_response_path, label, llm_client):
    # A fixed, label-independent prior keeps OVON runs reproducible without Ollama.
    # It does not use episode annotations or infer a room from the target label.
    if llm_client.llm_client == "none":
        return [], "everywhere", 0.5

    with open(llm_answer_path, "a+") as answer_file:
        answer_file.seek(0)
        for line in answer_file:
            prefix = f"{label}:"
            if line.startswith(prefix):
                return _parse_cached_answer(line[len(prefix):].strip(), label)

        # Import optional LLM dependencies only when a missing cache requires them.
        from llm.answer import get_answer

        answer, response = get_answer(prompt=label, client=llm_client)
        related, room, threshold = _parse_cached_answer(repr(answer), label)
        answer_file.write(f"\n{label}: {answer}")
        with open(llm_response_path, "a+") as response_file:
            response_file.write(f"\n{label}: {response}")
        return related, room, threshold
