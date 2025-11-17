"""
Détection d'outils dans les transcriptions d'entretiens
- Traite 3 versions de CSV (avec différents niveaux de granularité)
- Détecte les outils avec NER (CamemBERT)
- Génère 3 CSV avec colonnes: filename, text, word_count, tools, tool_number
- Génère 1 dictionnaire global des outils: tool, number_of_time_mentioned
"""

from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from collections import Counter


class ToolDetector:
    """Détecte les outils dans les textes d'artisans"""
    
    def __init__(self):
        """Initialiser le modèle NER"""
        print("Chargement du modele NER CamemBERT...")
        self.ner_pipeline = pipeline(
            "ner",
            model="cmarkea/distilcamembert-base-ner",
            aggregation_strategy="simple"
        )
        print("Modele charge avec succes.\n")
    
    def detect_tools_in_text(self, text: str) -> List[str]:
        """
        Detecter tous les outils dans un texte
        
        Args:
            text: Texte a analyser
        
        Returns:
            Liste des outils trouves (peut contenir des doublons)
        """
        if not text or pd.isna(text):
            return []
        
        try:
            # Extraire les entites avec NER
            entities = self.ner_pipeline(text)
            
            # Filtrer pour garder uniquement les entites de type TOOL ou objet
            tools = []
            for entity in entities:
                entity_type = entity.get("entity_group", "")
                # Garder TOOL
                if entity_type in ["TOOL","OUTIL"]:
                    tool_name = entity["word"].strip()
                    if tool_name:
                        tools.append(tool_name)
            
            return tools
            
        except Exception as e:
            print(f"Erreur lors de la detection: {e}")
            return []
    
    def process_csv(self, input_csv: Path, output_csv: Path) -> Tuple[pd.DataFrame, Counter]:
        """
        Traiter un fichier CSV et ajouter colonnes tools et tool_number
        
        Args:
            input_csv: Chemin du CSV d'entree
            output_csv: Chemin du CSV de sortie
        
        Returns:
            Tuple (DataFrame traite, Counter des outils)
        """
        # Charger le CSV
        df = pd.read_csv(input_csv)
        
        # Verifier les colonnes requises
        required_cols = ['filename', 'text', 'word_count']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Le CSV doit contenir les colonnes: {required_cols}")
        
        # Initialiser les nouvelles colonnes
        tools_list = []
        tool_numbers = []
        all_tools_counter = Counter()
        
        # Traiter chaque ligne
        print(f"Traitement de {len(df)} lignes de {input_csv.name}...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Detection"):
            text = row['text']
            
            # Detecter les outils
            tools_found = self.detect_tools_in_text(text)
            
            # Mettre a jour le compteur global
            all_tools_counter.update(tools_found)
            
            # Preparer les valeurs pour les colonnes
            if tools_found:
                # Joindre les outils avec des virgules (peut avoir des doublons)
                tools_str = ", ".join(tools_found)
                tool_count = len(tools_found)
            else:
                # Pas d'outils trouves
                tools_str = ""
                tool_count = 0
            
            tools_list.append(tools_str)
            tool_numbers.append(tool_count)
        
        # Ajouter les colonnes au DataFrame
        df['tools'] = tools_list
        df['tool_number'] = tool_numbers
        
        # Sauvegarder le CSV de sortie
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding='utf-8')
        
        print(f"CSV sauvegarde: {output_csv}")
        print(f"Lignes avec outils: {(df['tool_number'] > 0).sum()}/{len(df)}")
        print(f"Total d'outils detectes: {sum(tool_numbers)}\n")
        
        return df, all_tools_counter
    
    def create_tool_dictionary(self, counter : Counter, output_path: Path):
        """
        Creer un dictionnaire global des outils a partir d un compteur
        
        Args:
            counters: counter des outils
            output_path: Chemin du CSV dictionnaire de sortie
        """
        
        # Creer un DataFrame
        dict_data = {
            'tool': [],
            'number_of_time_mentioned': []
        }
        
        # Trier par nombre de mentions (décroissant)
        for tool, count in counter.most_common():
            dict_data['tool'].append(tool)
            dict_data['number_of_time_mentioned'].append(count)
        
        df_dict = pd.DataFrame(dict_data)
        
        # Sauvegarder
        df_dict.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"\nDictionnaire des outils cree: {output_path}")
        print(f"Nombre d'outils uniques: {len(df_dict)}")
        print(f"Total de mentions: {df_dict['number_of_time_mentioned'].sum()}")
        
        return df_dict
    
    def process_all_csvs(self, input_csvs: List[Path], output_dir: Path, dict_filename: str = "tool_dictionary.csv"):
        """
        Traiter tous les CSV et generer le dictionnaire global
        
        Args:
            input_csvs: Liste des chemins des CSV d'entree
            output_dir: Dossier de sortie pour les CSV traites
            dict_filename: Nom du fichier dictionnaire
        """
        print("=" * 70)
        print("DETECTION D'OUTILS - TRAITEMENT DE PLUSIEURS CSV")
        print("=" * 70 + "\n")
        
        counters = []
        
        # Traiter chaque CSV
        for input_csv in input_csvs:
            if not input_csv.exists():
                print(f"ATTENTION: Le fichier {input_csv} n'existe pas. Passe.\n")
                continue
            
            # Generer le nom de sortie (meme nom dans le nouveau dossier)
            output_csv = output_dir / input_csv.name
            
            # Traiter
            _, counter = self.process_csv(input_csv, output_csv)
            counters.append(counter)
        
        # Creer le dictionnaire global
        if counters:
            dict_path = output_dir / dict_filename
            df_dict = self.create_tool_dictionary(counters[0], dict_path)
            
            # Afficher les 10 outils les plus mentionnes
            print("\nTop 10 des outils les plus mentionnes:")
            for idx, row in df_dict.head(10).iterrows():
                print(f"  {idx+1}. {row['tool']}: {row['number_of_time_mentioned']} mentions")
        
        print("\n" + "=" * 70)
        print("TRAITEMENT TERMINE")
        print("=" * 70)


if __name__ == "__main__":
    # Initialiser le detecteur
    detector = ToolDetector()
    
    # Definir les chemins des 3 CSV d'entree
    base_dir = Path("data/processed")
    
    input_csvs = [
        base_dir / "cleaned_paragraph.csv",  # Version paragraphe
        base_dir / "cleaned_sentence.csv",   # Version phrase
        base_dir / "cleaned_full.csv",       # Version document entier
    ]
    
    # Dossier de sortie
    output_dir = Path("data/processed_tool_ner")
    
    # Traiter tous les CSV et generer le dictionnaire
    detector.process_all_csvs(
        input_csvs=input_csvs,
        output_dir=output_dir,
        dict_filename="tool_dictionary.csv"
    )
