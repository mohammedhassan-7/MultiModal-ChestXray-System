"""MedGemma 4B VLM wrapper for report generation and QA."""
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import torch
from PIL import Image

from src.config import MEDGEMMA_ID, DEVICE, USE_4BIT


REPORT_PROMPT = (
    "You are a radiologist reading THIS specific chest X-ray. "
    "Describe ONLY what you actually see in this image - be specific and concrete, "
    "not generic.\n\n"
    "Write a structured report with two sections:\n"
    "FINDINGS: List specific observations. Cover:\n"
    "  - Any tubes/lines/devices and their positions (ETT, NG, central line, port, pacer).\n"
    "  - Lungs: opacities, consolidations, atelectasis, pleural effusion, pneumothorax, edema. "
    "Specify side (left/right) and lobe/zone.\n"
    "  - Heart size and mediastinal contours.\n"
    "  - Bones / soft tissues if notable.\n"
    "IMPRESSION: 1-3 numbered key takeaways for the clinician.\n\n"
    "If a finding is absent, say so explicitly (e.g., 'no pneumothorax'). "
    "Avoid the catch-all phrase 'no acute cardiopulmonary process' unless the image is "
    "genuinely unremarkable."
)


class MedGemmaVLM:
    def __init__(self, model_id: str = MEDGEMMA_ID):
        kwargs = {"dtype": torch.bfloat16}
        if USE_4BIT and DEVICE == "cuda":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        kwargs["device_map"] = DEVICE
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        self.model.eval()

    def _chat(self, image: Image.Image, text: str, max_new_tokens: int = 256,
              temperature: float | None = None) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = {"max_new_tokens": max_new_tokens}
        if temperature is not None:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = out[0, inputs["input_ids"].shape[-1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

    def chat_text(self, prompt: str, max_new_tokens: int = 384,
                  temperature: float | None = None, top_p: float = 0.9,
                  force_prefix: str = "") -> str:
        """Text-only chat (no image). `force_prefix` is appended to the assistant
        turn before generation — use e.g. '[' to make the model start emitting JSON
        immediately, skipping MedGemma 1.5's default 'thinking' mode.
        """
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        ) + force_prefix
        inputs = self.processor.tokenizer(text, return_tensors="pt").to(self.model.device)

        gen_kwargs = {"max_new_tokens": max_new_tokens}
        if temperature is not None:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = out[0, inputs["input_ids"].shape[-1]:]
        decoded = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return force_prefix + decoded

    def generate_report(self, image: Image.Image) -> str:
        # Light sampling counters MedGemma's tendency to default to the "normal"
        # template; longer budget lets it produce the full structured report.
        return self._chat(image, REPORT_PROMPT, max_new_tokens=400, temperature=0.4)

    def answer_question(self, image: Image.Image, question: str, context: str | None = None) -> str:
        if context:
            prompt = (
                "You are a radiologist examining the chest X-ray above. "
                "Below are reports from similar prior cases for reference only - do NOT "
                "quote them, do NOT say 'the report states' or 'the report indicates'. "
                "Answer the question as a direct clinical observation about THIS image, "
                "1-2 short sentences, image-grounded phrasing only.\n\n"
                f"Reference reports (do not quote):\n{context}\n\n"
                f"Question: {question}\n"
                "Answer:"
            )
        else:
            prompt = (
                "You are a radiologist examining the chest X-ray above. "
                "Answer in 1-2 short sentences as a direct clinical observation about this image.\n\n"
                f"Question: {question}\n"
                "Answer:"
            )
        return self._chat(image, prompt, max_new_tokens=180, temperature=0.3)
