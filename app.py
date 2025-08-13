##deploy to streamlit
import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
##load the model

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras import layers
class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, max_len, **kwargs):  # ✅ Accept **kwargs
        super().__init__(**kwargs)  # ✅ Pass kwargs to base Layer
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.pos_encoding.shape[-1],
            "max_len": self.pos_encoding.shape[1]
        })
        return config

model = tf.keras.models.load_model(
    "gpt_model.keras",
    custom_objects={"PositionalEncoding": PositionalEncoding}
)


# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Parameters
max_seq_length = 100
st.title("GPT Text Generation")
st.write("Enter a seed text to generate text:")
seed_text = st.text_input("Seed Text:")
def generate_text(seed_text, model, tokenizer, num_tokens=50, temperature=1.0):
    for _ in range(num_tokens):
        token_seq = tokenizer.texts_to_sequences([seed_text])[0]
        token_seq = token_seq[-max_seq_length:]  # Trim to max length
        padded_seq = pad_sequences([token_seq], maxlen=max_seq_length)

        preds = model.predict(padded_seq, verbose=0)[0, -1]  # Get prediction for last time step
        preds = np.asarray(preds).astype('float64')

        # Apply temperature sampling
        preds = np.log(preds + 1e-9) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))

        next_token_id = np.random.choice(len(preds), p=preds)
        next_word = tokenizer.index_word.get(next_token_id, '')

        seed_text += ' ' + next_word

        if next_word == '':  # Optional: break if OOV or unknown word
            break

    return seed_text
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def transformer_block(embed_dim, num_heads, ff_dim, dropout=0.1):
    inputs = layers.Input(shape=(None, embed_dim))
    
    # Multi-head self-attention
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attn_output = layers.Dropout(dropout)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    
    # Feed-forward network
    ffn = tf.keras.Sequential([
        layers.Dense(ff_dim, activation='relu'),
        layers.Dense(embed_dim)
    ])
    ffn_output = ffn(out1)
    ffn_output = layers.Dropout(dropout)(ffn_output)
    
    # Output after second residual connection + normalization
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    
    return Model(inputs=inputs, outputs=out2)



if st.button("Generate"):
    generated = generate_text(seed_text, model, tokenizer, num_tokens=50, temperature=1.0)
    st.write("📝 Generated Text:")
    st.write(generated)

    

