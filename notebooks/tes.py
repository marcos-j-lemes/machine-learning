"""
Mini Python AI Editor
─────────────────────
Coloque modelo_ide.pth no mesmo diretório e rode:
    python mini_ide.py

Se ainda não exportou o modelo, rode primeiro:
    python exportar_modelo.py
"""

import sys
import re
import threading
import torch
import torch.nn.functional as F

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPlainTextEdit, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QSplitter,
    QTreeWidget, QTreeWidgetItem, QTabWidget, QInputDialog,
    QMessageBox, QPushButton, QStatusBar,
)
from PySide6.QtGui import (
    QColor, QFont, QTextCharFormat, QTextCursor,
    QSyntaxHighlighter, QPainter, QFontMetrics, QPalette,
)
from PySide6.QtCore import Qt, QTimer, QRect, QSize, Signal, QObject

# ─────────────────────────────────────────────────────────
#  AJUSTE AQUI: importe sua arquitetura de modelo
# ─────────────────────────────────────────────────────────
# from seu_arquivo import SeuModelo
# ─────────────────────────────────────────────────────────

MODEL_PATH = "modelo_ide.pth"   # gerado pelo exportar_modelo.py


# =========================
# TOKENIZADOR SIMPLES
# (reconstruído a partir do vocab salvo)
# =========================

class SimpleTokenizer:
    def __init__(self, word_to_idx, idx_to_word):
        self.word_to_idx = word_to_idx
        self.idx_to_word = {int(k): v for k, v in idx_to_word.items()}
        self.unk = word_to_idx.get("<UNK>", 0)

    def encode(self, text):
        tokens = text.split()
        return [self.word_to_idx.get(t, self.unk) for t in tokens]

    def decode(self, ids):
        words = [self.idx_to_word.get(i, "<UNK>") for i in ids]
        return " ".join(w for w in words if w not in ("<BOS>", "<EOS>", "<PAD>"))


# =========================
# MOTOR DE INFERÊNCIA
# =========================

class InferenceEngine(QObject):
    """
    Carrega o modelo e roda generate() numa thread separada
    para não travar a UI.
    """
    prediction_ready = Signal(str)
    status_changed   = Signal(str)

    def __init__(self):
        super().__init__()
        self.model     = None
        self.tokenizer = None
        self.seq_len   = 128
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self._running  = False

    def load(self, path=MODEL_PATH):
        try:
            ckpt = torch.load(path, map_location=self.device)

            self.tokenizer = SimpleTokenizer(
                ckpt["word_to_idx"],
                ckpt["idx_to_word"],
            )
            self.seq_len = ckpt.get("seq_len", 128)

            # ── Instancie sua arquitetura aqui ──────────────────
            # vocab_size = ckpt["vocab_size"]
            # self.model = SeuModelo(vocab_size=vocab_size, ...)
            # self.model.load_state_dict(ckpt["state_dict"])
            # self.model.to(self.device)
            # self.model.eval()
            # ────────────────────────────────────────────────────

            # Placeholder enquanto arquitetura não é conectada:
            self.model = None

            self.status_changed.emit(
                f"✓ Modelo carregado  |  vocab={len(self.tokenizer.word_to_idx)}  "
                f"|  device={self.device.upper()}"
            )
            return True

        except FileNotFoundError:
            self.status_changed.emit(
                f"⚠ '{path}' não encontrado — usando sugestões estáticas"
            )
            return False
        except Exception as e:
            self.status_changed.emit(f"⚠ Erro ao carregar modelo: {e}")
            return False

    # ─── Inferência real ────────────────────────────────

    def _generate(self, prompt, max_new_tokens=12, temperature=0.8,
                  top_k=40, top_p=0.95):
        if self.model is None or self.tokenizer is None:
            return _static_predict(prompt)

        self.model.eval()
        initial_ids = self.tokenizer.encode(prompt)

        bos = self.tokenizer.word_to_idx.get("<BOS>")
        if bos is not None and (not initial_ids or initial_ids[0] != bos):
            initial_ids = [bos] + initial_ids

        input_ids = torch.tensor([initial_ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                seq = (input_ids if input_ids.shape[1] <= self.seq_len
                       else input_ids[:, -self.seq_len:])

                mask   = self.model.generate_causal_mask(seq.shape[1]).to(self.device)
                logits = self.model(seq, mask)
                next_logits = logits[:, -1, :] / temperature

                if top_k:
                    thresh = torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[next_logits < thresh] = -float("Inf")

                if top_p and top_p < 1.0:
                    sl, si = torch.sort(next_logits, descending=True)
                    cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                    remove = cp > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    next_logits[si[remove]] = -float("Inf")

                probs   = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, 1)
                word    = self.tokenizer.idx_to_word.get(next_id.item(), "<UNK>")

                if word in ("<EOS>", "<PAD>"):
                    break

                input_ids = torch.cat([input_ids, next_id], dim=1)

        ids = input_ids[0].tolist()
        if bos is not None and ids and ids[0] == bos:
            ids = ids[1:]
        return self.tokenizer.decode(ids)

    def predict_async(self, prompt, callback,
                      max_new_tokens=12, temperature=0.8,
                      top_k=40, top_p=0.95):
        if self._running:
            return
        self._running = True

        def _run():
            try:
                full = self._generate(prompt, max_new_tokens,
                                      temperature, top_k, top_p)
                # Devolve só os tokens novos
                new_tokens = full[len(prompt):].strip() if full.startswith(prompt) else full
                self.prediction_ready.emit(new_tokens)
            finally:
                self._running = False

        threading.Thread(target=_run, daemon=True).start()


# =========================
# SUGESTÕES ESTÁTICAS
# (fallback sem modelo)
# =========================

def _static_predict(text):
    last = text.split("\n")[-1].strip()
    table = {
        "def soma(a, b):": "\n    return a + b",
        "for":    " i in range(10):",
        "if":     " True:",
        "class":  " MinhaClasse:",
        "def":    " funcao():",
        "return": " None",
        "import": " os",
        "print(": '"Hello, World!")',
    }
    for k, v in table.items():
        if last == k or last.endswith(k):
            return v
    return ""


# =========================
# CORES / TEMA
# =========================

THEME = {
    "bg0":     "#0d1117",
    "bg1":     "#161b22",
    "bg2":     "#1c2128",
    "bg3":     "#21262d",
    "bg4":     "#2d333b",
    "border":  "#30363d",
    "text":    "#e6edf3",
    "muted":   "#7d8590",
    "accent":  "#388bfd",
    "accent2": "#58a6ff",
    "green":   "#3fb950",
    "orange":  "#d29922",
    "pink":    "#ff7b72",
    "purple":  "#d2a8ff",
    "teal":    "#79c0ff",
    "yellow":  "#e3b341",
    "ghost":   "#58a6ff",
    "loading": "#f78166",
}

KEYWORDS = {
    "False","None","True","and","as","assert","async","await","break",
    "class","continue","def","del","elif","else","except","finally",
    "for","from","global","if","import","in","is","lambda","nonlocal",
    "not","or","pass","raise","return","try","while","with","yield"
}
BUILTINS = {
    "print","len","range","int","str","float","list","dict","tuple",
    "set","bool","type","isinstance","hasattr","getattr","setattr",
    "super","property","staticmethod","classmethod","enumerate","zip",
    "map","filter","sorted","reversed","sum","min","max","abs","round",
    "open","input","format","repr","append","extend","insert","remove",
    "pop","update"
}


# =========================
# SYNTAX HIGHLIGHTER
# =========================

class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self._rules = self._build_rules()

    def _fmt(self, color, bold=False, italic=False):
        f = QTextCharFormat()
        f.setForeground(QColor(color))
        if bold:   f.setFontWeight(700)
        if italic: f.setFontItalic(True)
        return f

    def _build_rules(self):
        t = THEME
        rules = []
        for p in [r'"""[\s\S]*?"""', r"'''[\s\S]*?'''",
                  r'f"[^"\n]*"', r"f'[^'\n]*'",
                  r'"[^"\n]*"',  r"'[^'\n]*'"]:
            rules.append((re.compile(p), self._fmt(t["green"])))
        rules.append((re.compile(r'#[^\n]*'),           self._fmt(t["muted"], italic=True)))
        rules.append((re.compile(r'\b\d+(\.\d+)?\b'),   self._fmt(t["yellow"])))
        rules.append((re.compile(r'@\w+'),               self._fmt(t["orange"])))
        kw = r'\b(' + '|'.join(re.escape(k) for k in KEYWORDS) + r')\b'
        rules.append((re.compile(kw), self._fmt(t["pink"], bold=True)))
        bi = r'\b(' + '|'.join(re.escape(b) for b in BUILTINS) + r')\b'
        rules.append((re.compile(bi), self._fmt(t["teal"])))
        rules.append((re.compile(r'\b(self|cls)\b'),    self._fmt(t["orange"])))
        rules.append((re.compile(r'(?<=def )\w+'),      self._fmt(t["purple"])))
        rules.append((re.compile(r'(?<=class )\w+'),    self._fmt(t["yellow"], bold=True)))
        return rules

    def highlightBlock(self, text):
        for pattern, fmt in self._rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), fmt)


# =========================
# NUMERAÇÃO DE LINHAS
# =========================

class LineNumberArea(QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self):
        return QSize(self.editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self.editor.line_number_area_paint_event(event)


# =========================
# EDITOR
# =========================

class CodeEditor(QPlainTextEdit):

    ghost_changed = Signal(str)

    def __init__(self, engine: InferenceEngine):
        super().__init__()
        self.engine     = engine
        self.ghost_text = ""
        self.is_loading = False
        self._setup_appearance()

        self.line_number_area = LineNumberArea(self)
        self.blockCountChanged.connect(self.update_line_number_area_width)
        self.updateRequest.connect(self.update_line_number_area)
        self.update_line_number_area_width(0)

        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._request_prediction)
        self.textChanged.connect(lambda: self._timer.start(350))

        self.engine.prediction_ready.connect(self._on_prediction)

    def _setup_appearance(self):
        t = THEME
        self.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: {t['bg0']};
                color: {t['text']};
                border: none;
                selection-background-color: {t['accent']};
            }}
        """)
        font = QFont("JetBrains Mono", 12)
        font.setStyleHint(QFont.StyleHint.Monospace)
        if not font.exactMatch():
            font = QFont("Fira Code", 12)
        if not font.exactMatch():
            font = QFont("Courier New", 12)
        self.setFont(font)
        self.setTabStopDistance(QFontMetrics(font).horizontalAdvance(' ') * 4)

    # ─── Line numbers ────────────────────────────────

    def line_number_area_width(self):
        digits = max(1, len(str(self.blockCount())))
        return 10 + self.fontMetrics().horizontalAdvance('9') * digits + 16

    def update_line_number_area_width(self, _=0):
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def update_line_number_area(self, rect, dy):
        if dy:
            self.line_number_area.scroll(0, dy)
        else:
            self.line_number_area.update(0, rect.y(),
                                         self.line_number_area.width(),
                                         rect.height())
        if rect.contains(self.viewport().rect()):
            self.update_line_number_area_width(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.line_number_area.setGeometry(
            QRect(cr.left(), cr.top(),
                  self.line_number_area_width(), cr.height()))

    def line_number_area_paint_event(self, event):
        t = THEME
        painter = QPainter(self.line_number_area)
        painter.fillRect(event.rect(), QColor(t["bg1"]))
        block  = self.firstVisibleBlock()
        bn     = block.blockNumber()
        top    = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())
        cur    = self.textCursor().blockNumber()
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.setPen(QColor(t["text"] if bn == cur else t["muted"]))
                painter.drawText(0, top,
                                 self.line_number_area.width() - 8,
                                 self.fontMetrics().height(),
                                 Qt.AlignmentFlag.AlignRight, str(bn + 1))
            block  = block.next()
            top    = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            bn    += 1

    # ─── Predição ────────────────────────────────────

    def _request_prediction(self):
        prompt = self.toPlainText()
        if not prompt.strip():
            return
        self.is_loading = True
        self.ghost_text = ""
        self.viewport().update()
        self.ghost_changed.emit("")
        self.engine.predict_async(prompt, None)

    def _on_prediction(self, text):
        self.is_loading = False
        self.ghost_text = text
        self.ghost_changed.emit(text)
        self.viewport().update()

    # ─── Key events ──────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key.Key_Tab:
            if self.ghost_text:
                cursor  = self.textCursor()
                current = self.toPlainText()
                sep = "" if (not current or current[-1] in (" ", "\n", "(", "[", "{")) else " "
                cursor.insertText(sep + self.ghost_text)
                self.ghost_text = ""
                self.viewport().update()
                self.ghost_changed.emit("")
            else:
                self.textCursor().insertText("    ")
            return

        if key == Qt.Key.Key_Escape:
            self.ghost_text = ""
            self.is_loading = False
            self.viewport().update()
            self.ghost_changed.emit("")
            return

        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            cursor   = self.textCursor()
            line     = cursor.block().text()
            indent_m = re.match(r'^(\s*)', line)
            indent   = indent_m.group(1) if indent_m else ""
            if line.rstrip().endswith(":"):
                indent += "    "
            self.ghost_text = ""
            super().keyPressEvent(event)
            self.textCursor().insertText(indent)
            return

        self.ghost_text = ""
        self.is_loading = False
        self.viewport().update()
        super().keyPressEvent(event)

    # ─── Ghost text rendering ─────────────────────────

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self.viewport())
        fm      = self.fontMetrics()
        cursor  = self.cursorRect()
        x       = cursor.x()
        y       = cursor.top() + fm.ascent()

        if self.is_loading:
            painter.setPen(QColor(THEME["loading"]))
            painter.setOpacity(0.6)
            painter.drawText(x + 4, y, "  ⟳ gerando…")

        elif self.ghost_text:
            painter.setPen(QColor(THEME["ghost"]))
            painter.setOpacity(0.45)
            lines = self.ghost_text.split("\n")
            lh    = fm.height()
            for i, line in enumerate(lines):
                draw_x = x if i == 0 else (self.line_number_area.width() + 4)
                painter.drawText(draw_x, y + i * lh, line)

        painter.end()


# =========================
# PAINEL DE ARQUIVOS
# =========================

class FilePanel(QWidget):
    file_selected = Signal(str)
    file_created  = Signal(str)
    file_deleted  = Signal(str)

    def __init__(self):
        super().__init__()
        t = THEME
        self.setStyleSheet(f"""
            QWidget     {{ background: {t['bg1']}; }}
            QTreeWidget {{ background: {t['bg1']}; color: {t['text']};
                           border: none; font-size: 12px; }}
            QTreeWidget::item {{ padding: 3px 6px; }}
            QTreeWidget::item:selected {{ background: {t['bg4']}; color: {t['accent2']}; }}
            QTreeWidget::item:hover    {{ background: {t['bg3']}; }}
            QPushButton {{ background: transparent; color: {t['muted']};
                           border: none; font-size: 15px; padding: 2px 6px; }}
            QPushButton:hover {{ color: {t['text']}; background: {t['bg3']};
                                 border-radius: 3px; }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QWidget()
        header.setFixedHeight(32)
        header.setStyleSheet(f"border-bottom: 1px solid {t['border']};")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(8, 0, 4, 0)
        lbl = QLabel("EXPLORADOR")
        lbl.setStyleSheet(f"color:{t['muted']}; font-size:10px; font-weight:bold; letter-spacing:1px;")
        hl.addWidget(lbl)
        hl.addStretch()
        for text, slot, tip in [("+", self._create, "Novo"), ("−", self._delete, "Excluir")]:
            b = QPushButton(text)
            b.setFixedSize(22, 22)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            hl.addWidget(b)
        layout.addWidget(header)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(lambda item: self.file_selected.emit(item.text(0)))
        layout.addWidget(self.tree)

    def populate(self, names, active=None):
        self.tree.clear()
        for n in names:
            item = QTreeWidgetItem([n])
            self.tree.addTopLevelItem(item)
            if n == active:
                self.tree.setCurrentItem(item)

    def _create(self):
        name, ok = QInputDialog.getText(self, "Novo arquivo", "Nome (.py):")
        if ok and name:
            if not name.endswith(".py"): name += ".py"
            self.file_created.emit(name)

    def _delete(self):
        item = self.tree.currentItem()
        if not item: return
        n = item.text(0)
        if QMessageBox.question(self, "Excluir", f"Excluir '{n}'?",
                                QMessageBox.StandardButton.Yes |
                                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self.file_deleted.emit(n)


# =========================
# PAINEL DE SUGESTÃO
# =========================

class SuggestionPanel(QWidget):
    """Mostra a última sugestão do modelo de forma legível."""

    def __init__(self):
        super().__init__()
        t = THEME
        self.setFixedHeight(52)
        self.setStyleSheet(f"background: {t['bg2']}; border-top: 1px solid {t['border']};")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)

        icon = QLabel("🤖")
        icon.setFixedWidth(24)
        layout.addWidget(icon)

        col = QVBoxLayout()
        col.setSpacing(1)
        lbl_top = QLabel("Sugestão do modelo")
        lbl_top.setStyleSheet(f"color: {t['muted']}; font-size: 10px;")
        self.lbl_suggestion = QLabel("—")
        self.lbl_suggestion.setStyleSheet(
            f"color: {t['accent2']}; font-size: 12px; font-family: monospace;"
        )
        self.lbl_suggestion.setWordWrap(True)
        col.addWidget(lbl_top)
        col.addWidget(self.lbl_suggestion)
        layout.addLayout(col)

        hint = QLabel("TAB aceita  ·  ESC descarta")
        hint.setStyleSheet(f"color: {t['muted']}; font-size: 10px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(hint)

    def update_suggestion(self, text):
        if text:
            display = text.replace("\n", "↵ ").strip()
            if len(display) > 90:
                display = display[:87] + "…"
            self.lbl_suggestion.setText(display)
        else:
            self.lbl_suggestion.setText("—")


# =========================
# JANELA PRINCIPAL
# =========================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python AI Editor")
        self.resize(1200, 760)

        self.files = {
            "main.py":   'def main():\n    print("Olá, mundo!")\n\nif __name__ == "__main__":\n    main()\n',
            "modelo.py": '# Seu modelo aqui\n\ndef predict_next(text):\n    last = text.split("\\n")[-1].strip()\n    return ""\n',
            "utils.py":  'def soma(a, b):\n    return a + b\n\ndef fatorial(n):\n    if n <= 1:\n        return 1\n    return n * fatorial(n - 1)\n',
        }
        self.active_file = "main.py"
        self.engine = InferenceEngine()

        self._apply_theme()
        self._build_ui()

        # Carrega modelo em background para não travar a abertura
        threading.Thread(target=lambda: self.engine.load(MODEL_PATH), daemon=True).start()

    def _apply_theme(self):
        t = THEME
        self.setStyleSheet(f"""
            QMainWindow  {{ background: {t['bg0']}; }}
            QSplitter::handle {{ background: {t['border']}; width: 1px; }}
            QTabWidget::pane  {{ border: none; background: {t['bg0']}; }}
            QTabBar::tab {{
                background: {t['bg1']}; color: {t['muted']};
                padding: 6px 16px; border: none;
                border-right: 1px solid {t['border']}; font-size: 12px;
            }}
            QTabBar::tab:selected {{ background: {t['bg0']}; color: {t['text']};
                                     border-bottom: 2px solid {t['accent']}; }}
            QTabBar::tab:hover    {{ color: {t['text']}; }}
            QStatusBar  {{ background: {t['bg1']}; color: {t['muted']};
                           font-size: 11px; border-top: 1px solid {t['border']}; }}
            QStatusBar QLabel {{ color: {t['muted']}; padding: 0 8px; }}
        """)

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        self.file_panel = FilePanel()
        self.file_panel.setMinimumWidth(160)
        self.file_panel.setMaximumWidth(260)
        self.file_panel.file_selected.connect(self._load_file)
        self.file_panel.file_created.connect(self._create_file)
        self.file_panel.file_deleted.connect(self._delete_file)
        splitter.addWidget(self.file_panel)

        right = QWidget()
        right.setStyleSheet(f"background: {THEME['bg0']};")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        rl.addWidget(self.tabs)

        self.suggestion_panel = SuggestionPanel()
        rl.addWidget(self.suggestion_panel)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([200, 900])

        self.setCentralWidget(splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.lbl_model = QLabel("⟳ Carregando modelo…")
        self.lbl_lines = QLabel("1 linha")
        self.lbl_pos   = QLabel("Ln 1, Col 1")
        self.status_bar.addWidget(self.lbl_model)
        self.status_bar.addWidget(self.lbl_lines)
        self.status_bar.addPermanentWidget(self.lbl_pos)

        self.engine.status_changed.connect(self.lbl_model.setText)

        self._refresh_sidebar()
        self._load_file(self.active_file)

    # ─── Arquivos ─────────────────────────────────────

    def _refresh_sidebar(self):
        self.file_panel.populate(list(self.files.keys()), self.active_file)

    def _load_file(self, name):
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == name:
                self.tabs.setCurrentIndex(i)
                self.active_file = name
                self._refresh_sidebar()
                return

        editor = CodeEditor(self.engine)
        PythonHighlighter(editor.document())
        editor.setPlainText(self.files.get(name, ""))
        editor.textChanged.connect(lambda: self._sync(name, editor))
        editor.cursorPositionChanged.connect(lambda: self._update_pos(editor))
        editor.ghost_changed.connect(self.suggestion_panel.update_suggestion)

        self.tabs.addTab(editor, name)
        self.tabs.setCurrentWidget(editor)
        self.active_file = name
        self._refresh_sidebar()

    def _sync(self, name, editor):
        self.files[name] = editor.toPlainText()
        lines = editor.blockCount()
        self.lbl_lines.setText(f"{lines} linha{'s' if lines != 1 else ''}")

    def _update_pos(self, editor):
        cur = editor.textCursor()
        self.lbl_pos.setText(f"Ln {cur.blockNumber()+1}, Col {cur.columnNumber()+1}")

    def _on_tab_changed(self, index):
        if index < 0: return
        self.active_file = self.tabs.tabText(index)
        self._refresh_sidebar()

    def _close_tab(self, index):
        if self.tabs.count() <= 1: return
        self.tabs.removeTab(index)
        self.active_file = self.tabs.tabText(self.tabs.currentIndex())
        self._refresh_sidebar()

    def _create_file(self, name):
        if name not in self.files:
            self.files[name] = f"# {name}\n"
        self._load_file(name)

    def _delete_file(self, name):
        if len(self.files) <= 1:
            QMessageBox.warning(self, "Aviso", "Não é possível excluir o único arquivo.")
            return
        del self.files[name]
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == name:
                self.tabs.removeTab(i)
                break
        if self.active_file == name:
            self.active_file = list(self.files.keys())[0]
            self._load_file(self.active_file)
        self._refresh_sidebar()


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    for role, color in [
        (QPalette.ColorRole.Window,          THEME["bg0"]),
        (QPalette.ColorRole.WindowText,      THEME["text"]),
        (QPalette.ColorRole.Base,            THEME["bg0"]),
        (QPalette.ColorRole.AlternateBase,   THEME["bg1"]),
        (QPalette.ColorRole.Text,            THEME["text"]),
        (QPalette.ColorRole.Button,          THEME["bg2"]),
        (QPalette.ColorRole.ButtonText,      THEME["text"]),
        (QPalette.ColorRole.Highlight,       THEME["accent"]),
        (QPalette.ColorRole.HighlightedText, "#ffffff"),
    ]:
        palette.setColor(role, QColor(color))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())