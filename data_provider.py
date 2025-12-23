import pyodbc
import time
from config import Config

class DataProvider:
    def __init__(self):
        self.mock_db = {
            "4067898529187": {"po": "0900005032", "article": "ADIDAS_SHOE_TX", "size": "9.5", "qty": 50},
            "12345": {"po": "PO123", "article": "MOCK_SHOE", "size": "9", "qty": 10},
            "67890": {"po": "PO456", "article": "MOCK_BOOT", "size": "10", "qty": 5}
        }

        self.conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={Config.DB_SERVER};"
            f"DATABASE={Config.DB_NAME};"
            f"UID={Config.DB_USER};"
            f"PWD={Config.DB_PASS};"
            f"Connection Timeout=3;"
        )
        self.conn = None

    def _get_sql_connection(self):
        """Hàm helper để quản lý kết nối: Tự động reconnect nếu mất"""
        try:
            if self.conn is None:
                print("[SQL] Opening new connection...")
                self.conn = pyodbc.connect(self.conn_str)
            return self.conn
        except pyodbc.Error as e:
            print(f"[SQL CONNECT ERROR] {e}")
            self.conn = None
            return None

    def validate_po_barcode(self, po: str, barcode: str):
        if Config.USE_MOCK:
            return self._validate_mock(po, barcode)
        else:
            return self._validate_sql(po, barcode)

    def _validate_mock(self, po: str, barcode: str):
        print(f"[MOCK] Checking {barcode} in PO {po}")
        if barcode in self.mock_db:
            data = self.mock_db[barcode]
            if data["po"] == po:
                return {**data, "valid": True, "po_found": po, "bc_found": barcode}
        return None

    def _validate_sql(self, po: str, barcode: str):
        conn = self._get_sql_connection()
        if not conn:
            return None

        cursor = None
        try:
            cursor = conn.cursor()
            
            sql = """
            SELECT 
                Data_Shoebox_Detail.UPC, 
                Data_Shoebox_Detail.Size, 
                Data_Shoebox_Detail.PO, 
                Data_Shoebox_Detail.RY, 
                REPLACE(Data_Shoebox.Article, ' ', '') AS Article, 
                REPLACE(Data_Shoebox.Article, ' ', '') + '.bmp' AS Article_Image, 
                ISNULL(scbbss.Qty, 0) AS Quantity
            FROM Data_Shoebox_Detail
            LEFT JOIN Data_Shoebox 
                ON Data_Shoebox.RY_DDBH = Data_Shoebox_Detail.RY 
                AND Data_Shoebox.Size = Data_Shoebox_Detail.Size
            LEFT JOIN scbbss 
                ON scbbss.SCBH = Data_Shoebox_Detail.RY
                AND scbbss.XXCC = Data_Shoebox_Detail.Size
            WHERE Data_Shoebox_Detail.PO = ? AND Data_Shoebox_Detail.UPC = ?

            UNION ALL

            SELECT 
                Data_Shoebox_Detail.UPC, 
                Data_Shoebox_Detail.Size, 
                Data_Shoebox_Detail.PO, 
                Data_Shoebox_Detail.RY, 
                REPLACE(Data_Shoebox.Article, ' ', '') AS Article, 
                REPLACE(Data_Shoebox.Article, ' ', '') + '.bmp' AS Article_Image, 
                ISNULL(scbbss.Qty, 0) AS Quantity
            FROM Data_Shoebox_Detail
            LEFT JOIN Data_Shoebox 
                ON Data_Shoebox.RY_DDBH = Data_Shoebox_Detail.RY 
                AND Data_Shoebox.Size = Data_Shoebox_Detail.Size
            LEFT JOIN scbbss 
                ON scbbss.SCBH = Data_Shoebox_Detail.RY 
                AND scbbss.XXCC = Data_Shoebox_Detail.Size
            WHERE Data_Shoebox_Detail.PO = ? AND Data_Shoebox_Detail.UPC = ?
            """
            
            cursor.execute(sql, (po, barcode, po, barcode))
            row = cursor.fetchone()

            if row:
                return {
                    "valid": True,
                    "upc": row[0],
                    "size": row[1],
                    "po_found": row[2],
                    "ry": row[3],
                    "article": row[4],
                    "image": row[5],
                    "qty": row[6],
                    "bc_found": barcode
                }
            return None

        except pyodbc.Error as e:
            print(f"[SQL QUERY ERROR] {e}")
            self.conn = None 
            return None
        finally:
            if cursor:
                cursor.close()