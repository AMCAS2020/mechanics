import sys

from PyQt5 import QtWidgets
from leafi.leaf_interrogator_controller import LeafInterrogatorController

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.installEventFilter(app)
    LSA = LeafInterrogatorController()
    LSA.show()

    sys.exit(app.exec_())
