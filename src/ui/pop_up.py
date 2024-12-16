from PyQt6.QtWidgets import QLabel, QDialog, QVBoxLayout


class SimplePopup(QDialog):
    def __init__(self, message):
        super().__init__()
        self.setWindowTitle("Popup")
        self.setModal(True)
        self.setFocus()

        layout = QVBoxLayout()

        label = QLabel(message)
        layout.addWidget(label)

        self.setLayout(layout)
