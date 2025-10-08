# complexity_analyzer/tools_info.py
"""
Информация о поддерживаемых инструментах анализа.
"""

TOOLS_INFO = {
    'radon': {
        'name': 'Radon',
        'description': 'Cyclomatic Complexity анализатор',
        'metrics': ['Cyclomatic Complexity', 'Maintainability Index', 'Halstead metrics'],
        'install': 'pip install radon',
        'api_available': True,
        'url': 'https://radon.readthedocs.io/'
    },
    'lizard': {
        'name': 'Lizard',
        'description': 'Многоязычный анализатор сложности кода',
        'metrics': ['Cyclomatic Complexity', 'NLOC', 'Token Count', 'Parameter Count'],
        'install': 'pip install lizard',
        'api_available': True,
        'url': 'https://github.com/terryyin/lizard'
    },
    'mccabe': {
        'name': 'McCabe',
        'description': 'Классический McCabe Cyclomatic Complexity',
        'metrics': ['Cyclomatic Complexity'],
        'install': 'pip install mccabe',
        'api_available': True,
        'url': 'https://github.com/PyCQA/mccabe'
    },
    'complexipy': {
        'name': 'Complexipy',
        'description': 'Когнитивная сложность (написан на Rust)',
        'metrics': ['Cognitive Complexity'],
        'install': 'pip install complexipy',
        'api_available': True,
        'url': 'https://github.com/rohaquinlop/complexipy'
    },
    'halstead': {
        'name': 'Halstead Metrics',
        'description': 'Измеряет "интеллектуальное усилие" программирования',
        'metrics': ['Vocabulary', 'Volume', 'Difficulty', 'Effort', 'Time', 'Bugs'],
        'install': 'pip install multimetric',
        'api_available': True,
        'url': 'https://pypi.org/project/multimetric/'
    },
    'mi': {
        'name': 'Maintainability Index',
        'description': 'Композитная метрика maintainability (0-100)',
        'metrics': ['MI Score', 'MI Rank'],
        'install': 'pip install radon',
        'api_available': True,
        'url': 'https://radon.readthedocs.io/'
    },
    'nbd': {
        'name': 'Nested Block Depth',
        'description': 'Максимальная глубина вложенности блоков',
        'metrics': ['Max Depth'],
        'install': 'Built-in (uses AST)',
        'api_available': True,
        'url': 'https://docs.python.org/3/library/ast.html'
    }
}


def print_tools_info():
    """Выводит информацию о всех доступных инструментах"""
    print("="*80)
    print("ДОСТУПНЫЕ ИНСТРУМЕНТЫ АНАЛИЗА")
    print("="*80)
    
    for tool_id, info in TOOLS_INFO.items():
        print(f"\n📦 {info['name']} ({tool_id})")
        print(f"   {info['description']}")
        print(f"   Метрики: {', '.join(info['metrics'])}")
        print(f"   Установка: {info['install']}")
        print(f"   Python API: {'✅' if info['api_available'] else '❌'}")
        print(f"   URL: {info['url']}")
        if 'note' in info:
            print(f"   ⚠️  {info['note']}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    print_tools_info()
