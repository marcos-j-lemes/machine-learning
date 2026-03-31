### Baixar a biblioteca de logs

```bash
pip install logging
```

> agora usar no projeto:

```py
import logging

logging.basicConfig(filename="train.log", level=logging.INFO)

logging.info("Training started...")
```

### Padronização dos commit. 

```bash
pip install pre-commit
```

> escrever o yaml

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.0.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
```

> Para ativar:

```bash
pre-commit install
```

> para usar
`git commit -m "test"`

