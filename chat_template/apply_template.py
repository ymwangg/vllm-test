from fastchat.model.model_adapter import get_conversation_template

for model in ['llama-2','llama-3']:
    conv = get_conversation_template(model)
    conv.set_system_message("{system_message}")
    conv.append_message(conv.roles[0], "{user_prompt}")
    conv.append_message(conv.roles[1], "{response}")
    print(model)
    print(f"{conv.get_prompt()!r}")
