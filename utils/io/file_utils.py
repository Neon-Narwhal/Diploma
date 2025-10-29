"""Утилиты для работы с файлами"""
import json
import csv
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator
import logging

logger = logging.getLogger(__name__)

def read_source_file(file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
    """Чтение исходного кода из файла"""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {file_path}")
    
    if not path.suffix == '.py':
        logger.warning(f"File {file_path} is not a Python file")
    
    try:
        with open(path, 'r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        # Попробуем другие кодировки
        for alt_encoding in ['utf-8-sig', 'cp1251', 'latin1']:
            try:
                with open(path, 'r', encoding=alt_encoding) as f:
                    content = f.read()
                    logger.info(f"Successfully read {file_path} with encoding {alt_encoding}")
                    return content
            except UnicodeDecodeError:
                continue
        
        raise UnicodeDecodeError(f"Could not decode file {file_path} with any supported encoding")

def write_results(results: Any, output_path: Union[str, Path], 
                 format_type: str = 'json') -> None:
    """Запись результатов в файл"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format_type.lower() == 'json':
        write_json(results, path)
    elif format_type.lower() == 'csv':
        write_csv(results, path)
    elif format_type.lower() == 'pickle':
        write_pickle(results, path)
    else:
        raise ValueError(f"Unsupported format: {format_type}")

def write_json(data: Any, file_path: Union[str, Path], 
               indent: int = 2, ensure_ascii: bool = False) -> None:
    """Запись данных в JSON файл"""
    path = Path(file_path)
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, 
                     default=json_serializer, separators=(',', ': '))
    except Exception as e:
        logger.error(f"Failed to write JSON to {file_path}: {e}")
        raise

def read_json(file_path: Union[str, Path]) -> Any:
    """Чтение данных из JSON файла"""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to read JSON from {file_path}: {e}")
        raise

def write_csv(data: Union[List[Dict], List[List]], file_path: Union[str, Path],
              headers: Optional[List[str]] = None, delimiter: str = ',') -> None:
    """Запись данных в CSV файл"""
    path = Path(file_path)
    
    if not data:
        logger.warning(f"No data to write to {file_path}")
        return
    
    try:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            if isinstance(data[0], dict):
                # Данные в виде словарей
                fieldnames = headers or list(data[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
                writer.writeheader()
                writer.writerows(data)
            else:
                # Данные в виде списков
                writer = csv.writer(f, delimiter=delimiter)
                if headers:
                    writer.writerow(headers)
                writer.writerows(data)
    except Exception as e:
        logger.error(f"Failed to write CSV to {file_path}: {e}")
        raise

def read_csv(file_path: Union[str, Path], delimiter: str = ',', 
             has_header: bool = True) -> List[Dict[str, str]]:
    """Чтение данных из CSV файла"""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            if has_header:
                reader = csv.DictReader(f, delimiter=delimiter)
                return list(reader)
            else:
                reader = csv.reader(f, delimiter=delimiter)
                return [{'col_' + str(i): val for i, val in enumerate(row)} 
                       for row in reader]
    except Exception as e:
        logger.error(f"Failed to read CSV from {file_path}: {e}")
        raise

def write_pickle(data: Any, file_path: Union[str, Path]) -> None:
    """Запись данных в pickle файл"""
    path = Path(file_path)
    
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logger.error(f"Failed to write pickle to {file_path}: {e}")
        raise

def read_pickle(file_path: Union[str, Path]) -> Any:
    """Чтение данных из pickle файла"""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to read pickle from {file_path}: {e}")
        raise

def json_serializer(obj: Any) -> Any:
    """Сериализатор для JSON (обработка специальных типов)"""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, 'name') and hasattr(obj, 'value'):  # Enum objects
        return obj.value
    else:
        return str(obj)

def find_python_files(directory: Union[str, Path], 
                     recursive: bool = True,
                     exclude_patterns: Optional[List[str]] = None) -> List[Path]:
    """Поиск Python файлов в директории"""
    directory = Path(directory)
    exclude_patterns = exclude_patterns or ['__pycache__', '.git', '.pytest_cache', 'venv', '.venv']
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    python_files = []
    
    if recursive:
        pattern = "**/*.py"
    else:
        pattern = "*.py"
    
    for file_path in directory.glob(pattern):
        # Проверяем, не попадает ли файл под исключения
        should_exclude = False
        for exclude_pattern in exclude_patterns:
            if exclude_pattern in str(file_path):
                should_exclude = True
                break
        
        if not should_exclude:
            python_files.append(file_path)
    
    return sorted(python_files)

def create_directory(directory: Union[str, Path], exist_ok: bool = True) -> Path:
    """Создание директории"""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=exist_ok)
    return path

def backup_file(file_path: Union[str, Path], backup_suffix: str = '.bak') -> Path:
    """Создание резервной копии файла"""
    original_path = Path(file_path)
    
    if not original_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    backup_path = original_path.with_suffix(original_path.suffix + backup_suffix)
    
    import shutil
    shutil.copy2(original_path, backup_path)
    
    return backup_path

def safe_file_operation(operation_func, *args, backup: bool = True, **kwargs):
    """Безопасная операция с файлом (с созданием резервной копии)"""
    if backup and len(args) > 0:
        file_path = Path(args[0])
        if file_path.exists():
            backup_path = backup_file(file_path)
            logger.info(f"Created backup: {backup_path}")
    
    try:
        return operation_func(*args, **kwargs)
    except Exception as e:
        logger.error(f"File operation failed: {e}")
        if backup and len(args) > 0:
            backup_path = Path(args[0]).with_suffix(Path(args[0]).suffix + '.bak')
            if backup_path.exists():
                logger.info(f"Restoring from backup: {backup_path}")
                import shutil
                shutil.move(backup_path, args[0])
        raise

class FileIterator:
    """Итератор для обработки множества файлов"""
    
    def __init__(self, file_paths: List[Union[str, Path]], 
                 batch_size: Optional[int] = None):
        self.file_paths = [Path(p) for p in file_paths]
        self.batch_size = batch_size
        self.current_index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Union[Path, List[Path]]:
        if self.current_index >= len(self.file_paths):
            raise StopIteration
        
        if self.batch_size is None:
            # Возвращаем по одному файлу
            file_path = self.file_paths[self.current_index]
            self.current_index += 1
            return file_path
        else:
            # Возвращаем батч файлов
            batch = self.file_paths[self.current_index:self.current_index + self.batch_size]
            self.current_index += len(batch)
            return batch
    
    def progress(self) -> float:
        """Прогресс обработки (от 0 до 1)"""
        return min(self.current_index / len(self.file_paths), 1.0)

class ResultsWriter:
    """Класс для записи результатов в различных форматах"""
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def write_analysis_results(self, results: List[Dict[str, Any]], 
                             filename_base: str = "analysis_results"):
        """Запись результатов анализа в нескольких форматах"""
        # JSON для детальных результатов
        json_path = self.base_path / f"{filename_base}.json"
        write_json(results, json_path)
        
        # CSV для сводных данных
        if results:
            csv_data = []
            for result in results:
                csv_row = {
                    'file_path': result.get('file_path', ''),
                    'complexity_class': result.get('complexity_class', ''),
                    'confidence': result.get('confidence', 0),
                    'analyzer_name': result.get('analyzer_name', ''),
                    'analysis_time': result.get('analysis_time', 0),
                    'success': result.get('success', False)
                }
                csv_data.append(csv_row)
            
            csv_path = self.base_path / f"{filename_base}.csv"
            write_csv(csv_data, csv_path)
        
        # Сводный отчет
        summary = self._generate_summary(results)
        summary_path = self.base_path / f"{filename_base}_summary.json"
        write_json(summary, summary_path)
        
        return {
            'json_file': json_path,
            'csv_file': csv_path if results else None,
            'summary_file': summary_path
        }
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Генерация сводного отчета"""
        if not results:
            return {'total_files': 0, 'successful_analyses': 0}
        
        total_files = len(results)
        successful_analyses = sum(1 for r in results if r.get('success', False))
        
        # Распределение по классам сложности
        complexity_distribution = {}
        analysis_times = []
        confidence_scores = []
        
        for result in results:
            if result.get('success', False):
                complexity = result.get('complexity_class', 'unknown')
                complexity_distribution[complexity] = complexity_distribution.get(complexity, 0) + 1
                
                if 'analysis_time' in result:
                    analysis_times.append(result['analysis_time'])
                
                if 'confidence' in result:
                    confidence_scores.append(result['confidence'])
        
        summary = {
            'total_files': total_files,
            'successful_analyses': successful_analyses,
            'success_rate': successful_analyses / total_files if total_files > 0 else 0,
            'complexity_distribution': complexity_distribution,
            'avg_analysis_time': sum(analysis_times) / len(analysis_times) if analysis_times else 0,
            'avg_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        }
        
        return summary

def batch_process_files(file_paths: List[Union[str, Path]], 
                       process_func: callable,
                       batch_size: int = 10,
                       progress_callback: Optional[callable] = None) -> List[Any]:
    """Пакетная обработка файлов"""
    results = []
    
    iterator = FileIterator(file_paths, batch_size)
    
    for batch in iterator:
        batch_results = []
        
        for file_path in batch:
            try:
                result = process_func(file_path)
                batch_results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                batch_results.append({'error': str(e), 'file_path': str(file_path)})
        
        results.extend(batch_results)
        
        if progress_callback:
            progress_callback(iterator.progress())
    
    return results
