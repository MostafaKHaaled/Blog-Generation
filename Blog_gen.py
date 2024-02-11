
######################### packages that you need to install ##########################

# !pip install torch
# !pip install diffusers==0.10.2 transformers scipy ftfy accelerate
# !pip install accelerate
# !pip install git+https://github.com/huggingface/transformers.git
# !pip install gTTS
# !pip install --upgrade diffusers transformers accelerate peft





import streamlit as st
from gtts import gTTS
import io
import torch
import numpy as np
from PIL import Image
from torch import autocast
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline
from diffusers import LCMScheduler, AutoPipelineForText2Image

summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")


model_name = "gpt2" # Model size can be switched accordingly (e.g., "gpt2-medium")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
st.markdown(
"""
<html>
    <head>
    </head>
    <body>
        <h1><center><span style="color: #19acf6">Welcom in👋</span><span >ElectroPi Blog_generation 🕵️</span></center></h1>
        <br>
        <br>
    </body>
</html>
""",
unsafe_allow_html=True)


user_text = st.text_area("pleaser type any thing you want:")
prompt = user_text

def generate_text(prompt, max_length=1500, temperature=0.8):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


def generate_image(prompt, model_id="Lykon/dreamshaper-7", adapter_id="latent-consistency/lcm-lora-sdv1-5"):
    # Initialize AutoPipelineForText2Image
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    # Load and fuse lcm lora
    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()

    # Generate image using the provided prompt
    image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0).images[0]

    return image



if st.button("Generate Blog"):
    if user_text:
        generated_text = generate_text(prompt)
        language = 'en'
        tts = gTTS(text=generated_text, lang=language, slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        st.audio(audio_bytes, format="audio/mp3")
        st.write('---------------')
        st.subheader(f"this blog toke about {user_text}")
        st.write("\n");st.write("\n");st.write("\n");st.write("\n")

        split_text_generated = generated_text.strip().split('\n')
        counters = 0
        for i in split_text_generated:
            if i == '':
                continue
            else:
              col1, col2, col3= st.columns([5,2,5])
              counters = counters+1
              col1.write(i)
              col2.markdown(f"""<div style="display: flex; justify-content: center; align-items: center;"> <p>{counters}</p></div>""",unsafe_allow_html=True)
              imag = generate_image(prompt)
              col3.image(imag,use_column_width=True)
              st.markdown("---")
        st.markdown(f"conclusion")
        summary = summarizer(generated_text)[0]["summary_text"]
        language = 'en'
        tts = gTTS(text=summary, lang=language, slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        st.audio(audio_bytes, format="audio/mp3")
        st.markdown(f'''
        <div style="border: 3px solid #2fe738; padding: 10px;background-color: #000; color: white; padding: 10px;border-radius: 10px;font-size: 20px;">
        { summary }</div>
        ''', unsafe_allow_html=True)

        st.markdown("---")

        st.markdown(
                  """
                  <html>
                      <head>
                      </head>
                      <body>
                          <h1><center><span style="color: #19acf6">I Hope you Are Enjoy😍</span></center></h1>
                          <br>
                          <br>
                      </body>
                  </html>
                  """,
                  unsafe_allow_html=True)
    else:
        st.warning("Please enter some text.")