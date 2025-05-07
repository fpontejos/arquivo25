import argparse
import os
import sys
from typing import List

import chromadb
import numpy as np
import openai
import pandas as pd
import streamlit as st
import umap
from dotenv import load_dotenv
from openai import OpenAI
from pages.main_cols.chat import render_chat_column
from pages.main_cols.scatter import render_visualization_column
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.config import load_config, save_config
from utils.embeddings import OpenAIEmbedding
from utils.generator import OpenAIGenerator
from utils.retriever import ChromaDBRetriever
