import os
import torch
import torch.nn as nn
import requests
import numpy as np
import spacy
import yaml
import json
import networkx as nx
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from textstat import flesch_reading_ease
import nltk
from nltk.corpus import wordnet
import pandas as pd
from typing import Tuple

# Try to import Hugging Face datasets (used for huge training data)
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

# Configuration paths
CONFIG_PATH = "config.yaml"
JACET_PATH = "data/jacet8000/"
NEWS_CORPUS_PATH = "data/news_corpus/"
MODEL_PATH = "models/complexity_model.pt"
WORD_FREQ_PATH = "data/word_frequencies.txt"

# Setup directories
Path("data/jacet8000").mkdir(parents=True, exist_ok=True)
Path("data/news_corpus").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# Download NLTK resources
nltk.download('wordnet', quiet=True)

# Initialize spaCy (GPU will be used if available)
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

def setup_resources():
    """Download required resources automatically or create fallback files if needed."""
    
    # --- Auto-create config.yaml if it doesn't exist ---
    if not Path(CONFIG_PATH).exists():
        print("Creating default config.yaml ...")
        # Default config now includes a Hugging Face source for a huge dataset.
        default_config = {
            "sources": [
                {"type": "huggingface", "dataset": "wikitext", "config_name": "wikitext-103-raw-v1", "split": "train"},
                {"type": "jacet", "path": JACET_PATH},
                {"type": "newsela", "path": NEWS_CORPUS_PATH}
            ]
        }
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(default_config, f)
    
    resources = [
        {
            'name': 'Function words list',
            'url': 'https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt',
            'path': Path(NEWS_CORPUS_PATH) / "function_words.txt",
            'fallback': "the\nand\nof\nto\nin\nthat\nis\nit"
        },
        {
            'name': 'Word frequency data',
            'url': 'https://norvig.com/ngrams/count_1w.txt',
            'path': Path(WORD_FREQ_PATH),
            'fallback': "the\t10000\nand\t8000\nof\t7000\nto\t6000\nin\t5000"
        },
        {
            'name': 'Vocabulary levels data',
            'url': 'https://raw.githubusercontent.com/efcamlit/cefr-levels/master/cefr_wordlist.csv',  # May fail; fallback is provided
            'path': Path(JACET_PATH) / "cefr_words.csv",
            'fallback': "word,level\nthe,1\nbe,1\nto,1\nof,1\nand,1\na,1\nin,1\nthat,1\nhave,1\ni,1"
        },
        {
            'name': 'Reference TextRank scores',
            'url': 'https://raw.githubusercontent.com/selva86/datasets/master/reference_scores.json',  # Example URL; adjust if needed
            'path': Path(NEWS_CORPUS_PATH) / "reference_scores.json",
            'fallback': json.dumps({"mean": 0.15, "std": 0.03})
        }
    ]

    for res in resources:
        target_path = res['path']
        if not target_path.exists():
            try:
                print(f"Downloading {res['name']}...")
                response = requests.get(res['url'])
                response.raise_for_status()
                target_path.write_bytes(response.content)
                print(f"Downloaded {res['name']} to {target_path}")
                # Special handling for vocabulary levels: process the CSV into separate level files.
                if "cefr_words" in str(target_path):
                    _create_jacet_levels()
            except Exception as e:
                print(f"Failed to download {res['name']}: {e}")
                print(f"Writing fallback for {res['name']}")
                target_path.write_text(res['fallback'])
                if "cefr_words" in str(target_path):
                    _create_jacet_levels()

def _create_jacet_levels():
    """Convert the downloaded or fallback CEFR CSV data into JACET-style level files."""
    csv_path = Path(JACET_PATH) / "cefr_words.csv"
    if not csv_path.exists():
        print("CEFR CSV file not found; cannot create JACET levels.")
        return
    try:
        levels = defaultdict(list)
        with open(csv_path, encoding='utf-8') as fin:
            lines = fin.readlines()
            if len(lines) > 0:
                header = lines[0]
                for line in lines[1:]:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        word, level = parts[0], parts[1]
                        try:
                            levels[int(level)].append(word)
                        except Exception:
                            continue
        for level, words in levels.items():
            level_file = Path(JACET_PATH) / f"level_{level}.txt"
            with open(level_file, 'w', encoding='utf-8') as fout:
                fout.write('\n'.join(words))
        print("JACET levels created successfully.")
    except Exception as e:
        print("Error processing CEFR data, creating fallback JACET level 1:", e)
        _create_fallback_jacet()

def _create_fallback_jacet():
    """Create a minimal JACET level file as fallback."""
    fallback_words = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i']
    fallback_file = Path(JACET_PATH) / "level_1.txt"
    fallback_file.write_text("\n".join(fallback_words))
    print("Fallback JACET level file created.")

class JACETAnalyzer:
    def __init__(self):
        self.levels = self._load_jacet_levels()
        self.word_cache = defaultdict(int)
        
    def _load_jacet_levels(self):
        levels = defaultdict(set)
        try:
            for level_file in Path(JACET_PATH).glob("level_*.txt"):
                level = int(level_file.stem.split("_")[1])
                with open(level_file, encoding='utf-8') as f:
                    levels[level] = {word.strip().lower() for word in f if word.strip()}
            return levels
        except Exception as e:
            print("Error loading JACET levels, using minimal fallback:", e)
            return defaultdict(set, {1: {'the', 'be', 'to', 'of', 'and'}})

    def get_word_level(self, word: str) -> int:
        word = word.lower()
        if word in self.word_cache:
            return self.word_cache[word]
        for level in sorted(self.levels.keys(), reverse=True):
            if word in self.levels[level]:
                self.word_cache[word] = level
                return level
        self.word_cache[word] = 9  # Unknown words considered at level 9+
        return 9

class ComplexityAnalyzer:
    def __init__(self):
        setup_resources()
        self.jacet = JACETAnalyzer()
        self.function_words = self._load_function_words()
        self.ref_scores = self._load_reference_scores()
        self.word_freq = self._load_word_frequencies()
        self.a = 30  # threshold percentage
        
        # Initialize transformer models
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.sentence_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        
        # Device configuration (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sentence_model = self.sentence_model.to(self.device)
        self._init_complexity_model()

    def _init_complexity_model(self):
        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            print("Loaded existing complexity model.")

    def _load_function_words(self):
        try:
            with open(Path(NEWS_CORPUS_PATH) / "function_words.txt", encoding='utf-8') as f:
                return set(word.strip().lower() for word in f if word.strip())
        except Exception:
            return {'the', 'and', 'of', 'to', 'in'}

    def _load_reference_scores(self):
        try:
            with open(Path(NEWS_CORPUS_PATH) / "reference_scores.json", encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {'mean': 0.15, 'std': 0.03}

    def _load_word_frequencies(self):
        freq = defaultdict(float)
        try:
            with open(WORD_FREQ_PATH, encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        word, count = parts
                        freq[word.lower()] = float(count)
        except Exception:
            pass
        return freq

    def _textrank_sentence(self, sentence: str) -> float:
        words = [token.text.lower() for token in nlp(sentence) if token.text.lower() in self.function_words]
        graph = nx.DiGraph()
        window_size = 3
        for i in range(len(words)):
            for j in range(i+1, min(i+window_size, len(words))):
                if words[i] != words[j]:
                    graph.add_edge(words[i], words[j], weight=1/(j-i))
        try:
            scores = nx.pagerank(graph)
            return np.mean(list(scores.values())) if scores else 0.0
        except Exception:
            return 0.0

    def syntactic_complexity(self, text: str) -> float:
        doc = nlp(text)
        sentence_scores = [self._textrank_sentence(sent.text) for sent in doc.sents]
        essay_score = np.mean(sentence_scores) if sentence_scores else 0.0
        similarity = 1 - abs(essay_score - self.ref_scores['mean'])/self.ref_scores['std']
        return max(0.0, min(similarity, 1.0))

    def _get_replacements(self, word: str, author_level: int) -> dict:
        replacements = {'higher': [], 'lower': []}
        current_level = self.jacet.get_word_level(word)
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().lower().replace('_', ' ')
                if synonym == word:
                    continue
                syn_level = self.jacet.get_word_level(synonym)
                freq_weight = self.word_freq.get(synonym, 0.0)
                if syn_level > current_level and syn_level <= author_level + 2:
                    if freq_weight < 0.5:
                        replacements['higher'].append(synonym)
                elif syn_level < current_level and syn_level >= max(1, author_level - 2):
                    replacements['lower'].append(synonym)
        return replacements

    def lexical_complexity(self, text: str, author_level: int) -> Tuple[float, list]:
        doc = nlp(text)
        words = [token.text.lower() for token in doc if token.is_alpha]
        unique_words = list(set(words))
        C = len(unique_words)
        
        P = 0  # bonus credits (lower-level synonyms)
        N = 0  # penalty credits (higher-level synonyms)
        suggestions = []
        
        for word in unique_words:
            replacements = self._get_replacements(word, author_level)
            if replacements['higher']:
                N -= 1
                suggestions.append({
                    'word': word,
                    'replacements': replacements['higher'][:3],
                    'type': 'improvement'
                })
            if replacements['lower']:
                P += 1
        
        R = P + N
        c_times_a = C * (self.a / 100)
        if R >= c_times_a:
            f_r = c_times_a
        elif R <= -c_times_a:
            f_r = -c_times_a
        else:
            f_r = R
        try:
            sl = 1 + (f_r / c_times_a)
        except ZeroDivisionError:
            sl = 1.0
        return max(0.0, min(sl, 2.0)), suggestions

    def _get_embedding_features(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.sentence_model(**inputs)
        emb_scalar = outputs.last_hidden_state.mean().item()
        return emb_scalar

    def train(self, dataset_path: str, epochs=10, batch_size=32):
        dataset = ComplexityDataset(dataset_path)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        criterion = nn.BCELoss()
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for texts, levels, labels in loader:
                features = []
                for text, level in zip(texts, levels):
                    lc, _ = self.lexical_complexity(text, level)
                    sc = self.syntactic_complexity(text)
                    emb = self._get_embedding_features(text)
                    flesch = flesch_reading_ease(text) / 100
                    features.append([lc, sc, emb, flesch])
                X = torch.tensor(features, dtype=torch.float32).to(self.device)
                # Clamp target labels to the [0, 1] range.
                y = labels.to(self.device, dtype=torch.float32).clamp(0.0, 1.0)
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs.squeeze(), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")
            torch.save(self.model.state_dict(), MODEL_PATH)

    def analyze(self, text: str, author_level: int) -> dict:
        self.model.eval()
        lc, suggestions = self.lexical_complexity(text, author_level)
        sc = self.syntactic_complexity(text)
        emb = self._get_embedding_features(text)
        flesch = flesch_reading_ease(text) / 100
        with torch.no_grad():
            features = torch.tensor([[lc, sc, emb, flesch]], dtype=torch.float32).to(self.device)
            overall = self.model(features).item()
        return {
            "lexical": {"score": lc, "suggestions": suggestions},
            "syntactic": {"score": sc, "feedback": self._generate_syntactic_feedback(text)},
            "overall": overall,
            "readability": flesch_reading_ease(text),
            "embedding_feature": emb
        }

    def _generate_syntactic_feedback(self, text: str) -> list:
        doc = nlp(text)
        feedback = []
        for sent in doc.sents:
            structure = {"clauses": 0, "passive": 0, "complex_conjunctions": 0}
            for token in sent:
                if token.dep_ in ["cc", "mark"]:
                    structure["complex_conjunctions"] += 1
                if token.tag_ == "VBN" and "pass" in token.dep_:
                    structure["passive"] += 1
                if token.dep_ == "advcl":
                    structure["clauses"] += 1
            feedback.append({"sentence": sent.text, "structure": structure})
        return feedback

class ComplexityDataset(Dataset):
    def __init__(self, config_path):
        with open(config_path, encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.samples = []
        for source in self.config.get('sources', []):
            if source.get('type') == 'jacet':
                self._load_jacet_samples(source)
            elif source.get('type') == 'newsela':
                self._load_newsela_samples(source)
            elif source.get('type') == 'huggingface':
                self._load_huggingface_samples(source)
        if not self.samples:
            self._create_sample_data()

    def _load_jacet_samples(self, config):
        try:
            df = pd.read_csv(Path(config['path']) / 'annotations.csv')
            for _, row in df.iterrows():
                with open(Path(config['path']) / 'texts' / f"{row['id']}.txt", encoding='utf-8') as f:
                    text = f.read()
                self.samples.append((text, row['author_level'], row['complexity_score']))
        except Exception as e:
            print("Error loading JACET samples:", e)
            self._create_sample_data()

    def _load_newsela_samples(self, config):
        try:
            for level_dir in Path(config['path']).glob("level_*"):
                level = int(level_dir.name.split("_")[1])
                for text_file in level_dir.glob("*.txt"):
                    with open(text_file, encoding='utf-8') as f:
                        text = f.read()
                    self.samples.append((text, level, level/10))
        except Exception as e:
            print("Error loading Newsela samples:", e)
            self._create_sample_data()

    def _load_huggingface_samples(self, config):
        if load_dataset is None:
            print("Hugging Face datasets library not available.")
            return
        try:
            dataset = load_dataset(config.get("dataset"), config.get("config_name"), split=config.get("split", "train"))
            count = 0
            for sample in dataset:
                text = sample.get("text")
                if text is None:
                    continue
                # Compute the Flesch Reading Ease score, then clamp to [0,1]
                score = flesch_reading_ease(text) / 100.0
                score = max(0.0, min(score, 1.0))
                # Use an arbitrary author level (e.g., 6) for training purposes.
                self.samples.append((text, 6, score))
                count += 1
            print(f"Loaded {count} samples from Hugging Face dataset.")
        except Exception as e:
            print("Error loading Hugging Face samples:", e)
            self._create_sample_data()

    def _create_sample_data(self):
        samples = [
            ("The cat sat on the mat.", 3, 0.2),
            ("Sophisticated vocabulary enhances textual complexity.", 5, 0.8),
            ("Complex grammatical structures improve writing quality.", 6, 0.9)
        ]
        self.samples.extend(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, level, score = self.samples[idx]
        return text, level, score

if __name__ == "__main__":
    # This will automatically create config.yaml and download resources as needed.
    analyzer = ComplexityAnalyzer()
    
    # Train the model. (The Hugging Face source provides a huge dataset.)
    if not os.path.exists(MODEL_PATH):
        print("Training new model on a huge dataset (Hugging Face data)...")
        analyzer.train(CONFIG_PATH, epochs=10, batch_size=32)
    
    sample_text = ("The multifaceted interplay between sophisticated vocabulary and "
                   "complex grammatical structures necessitates comprehensive linguistic analysis.")
    result = analyzer.analyze(sample_text, author_level=6)
    print(json.dumps(result, indent=2))
