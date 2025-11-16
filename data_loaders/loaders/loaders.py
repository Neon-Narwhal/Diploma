"""Загрузчики датасетов из внешних источников"""
from datasets import load_dataset
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class HuggingFaceLoader:
    """Загрузка датасетов с HuggingFace"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("datasets/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_bigobench_complexity_labels(self, streaming: bool = True):
        """Загрузка меток сложности BigOBench"""
        logger.info("Загрузка complexity_labels_light.jsonl")
        
        dataset = load_dataset(
            "facebook/BigOBench",
            data_files="data/complexity_labels_light.jsonl",
            split="train",
            streaming=streaming,
            cache_dir=str(self.cache_dir)
        )
        
        return dataset
    
    def load_bigobench_solutions(self):
        """Загрузка кода решений BigOBench"""
        logger.info("Загрузка problem_and_human_solutions_list.jsonl")
        
        dataset = load_dataset(
            "facebook/BigOBench",
            data_files="data/problem_and_human_solutions_list.jsonl",
            split="train",
            streaming=True,
            cache_dir=str(self.cache_dir)
        )
        
        return dataset
    
    def load_bigobench_test_set(self, complexity_type: str = "time"):
        """Загрузка тестового набора"""
        if complexity_type not in ["time", "space"]:
            raise ValueError("complexity_type должен быть 'time' или 'space'")
        
        filename = f"data/{complexity_type}_complexity_test_set.jsonl"
        logger.info(f"Загрузка {filename}")
        
        dataset = load_dataset(
            "facebook/BigOBench",
            data_files=filename,
            split="train",
            cache_dir=str(self.cache_dir)
        )
        
        return dataset
    
    def build_solutions_index(self, solutions_dataset) -> Dict[str, Dict]:
        """Построение индекса solution_id -> данные из streaming датасета"""
        logger.info("Построение индекса решений из streaming датасета")
        
        solutions_map = {}
        processed_count = 0
        
        for problem in solutions_dataset:
            problem_id = problem['problem_id']
            problem_name = problem.get('problem_name', '')
            
            # correct_solution_list может быть None или пустым
            solution_list = problem.get('correct_solution_list')
            if not solution_list:
                continue
            
            for solution in solution_list:
                solution_id = solution.get('solution_id')
                solution_code = solution.get('solution_code')
                
                if not solution_id or not solution_code:
                    continue
                
                solutions_map[solution_id] = {
                    'code': solution_code,
                    'problem_id': problem_id,
                    'problem_name': problem_name
                }
            
            processed_count += 1
            if processed_count % 500 == 0:
                logger.info(f"Обработано {processed_count} задач, найдено {len(solutions_map)} решений")
        
        logger.info(f"Построен индекс для {len(solutions_map)} решений из {processed_count} задач")
        return solutions_map



class LocalLoader:
    """Загрузка из локальных файлов"""
    
    def __init__(self, data_dir: Path = Path("data/bigobench")):
        self.data_dir = data_dir
    
    def load_split(self, split: str = "train") -> List[Dict]:
        """Загрузка конкретного split"""
        filepath = self.data_dir / f"{split}.jsonl"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Файл {filepath} не найден")
        
        import json
        samples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line))
        
        return samples
    
    def load_metadata(self) -> Dict:
        """Загрузка метаданных"""
        metadata_path = self.data_dir / "metadata.json"
        
        if not metadata_path.exists():
            return {}
        
        import json
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
