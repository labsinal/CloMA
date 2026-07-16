"""
Qt model for displaying pandas DataFrames in a QTableView.
"""

from __future__ import annotations

import pandas as pd

from qtpy.QtCore import (
    Qt,
    QAbstractTableModel,
    QModelIndex,
)


class PandasModel(QAbstractTableModel):
    """
    Qt model that exposes a pandas DataFrame to a QTableView.

    Notes
    -----
    - Supports column sorting.
    - Displays floats with three decimal places.
    - Empty strings are shown for NaN values.
    - Keeps the original dataframe index, allowing external objects
      (such as napari layers) to remain synchronized after sorting.
    """

    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()

        # Internal dataframe used by the table
        self._df = dataframe.copy()

    ####################################################################
    # Required Qt methods

    def rowCount(self, parent=QModelIndex()) -> int:
        """Return number of rows."""

        return len(self._df)

    def columnCount(self, parent=QModelIndex()) -> int:
        """Return number of columns."""

        return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        """
        Return the value displayed in each cell.
        """

        if not index.isValid():
            return None

        value = self._df.iat[index.row(), index.column()]

        ###############################################################
        # Text shown in the table

        if role == Qt.DisplayRole:

            if pd.isna(value):
                return ""

            if isinstance(value, float):
                return f"{value:.3f}"

            return str(value)

        ###############################################################
        # Align numbers nicely

        if role == Qt.TextAlignmentRole:

            if isinstance(value, (int, float)):
                return Qt.AlignRight | Qt.AlignVCenter

            return Qt.AlignLeft | Qt.AlignVCenter

        return None

    ####################################################################
    # Headers

    def headerData(self, section, orientation, role):

        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])

        return str(section + 1)

    ####################################################################
    # Sorting

    def sort(self, column: int, order: Qt.SortOrder):
        """
        Sort dataframe by the selected column.
        """

        column_name = self._df.columns[column]

        ascending = order == Qt.AscendingOrder

        self.layoutAboutToBeChanged.emit()

        self._df = (
            self._df
            .sort_values(
                by=column_name,
                ascending=ascending,
                kind="mergesort",      # Stable sorting
            )
            .reset_index(drop=True)
        )

        self.layoutChanged.emit()

    ####################################################################
    # Convenience properties

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        Current dataframe displayed by the table.

        Note that this dataframe reflects the current sorting order.
        """

        return self._df

    def row(self, row: int) -> pd.Series:
        """
        Return one dataframe row.

        Parameters
        ----------
        row : int
            Row number in the current table.

        Returns
        -------
        pandas.Series
        """

        return self._df.iloc[row]

    def value(self, row: int, column: str):
        """
        Convenience method to access one value.

        Example
        -------
        >>> model.value(5, "area")
        """

        return self._df.iloc[row][column]

    ####################################################################
    # Updating

    def set_dataframe(self, dataframe: pd.DataFrame):
        """
        Replace the current dataframe.
        """

        self.beginResetModel()

        self._df = dataframe.copy()

        self.endResetModel()