import pandas as pd
import os

def xlsx_to_csv(directory):
    # Lặp qua mỗi file trong thư mục
    for filename in os.listdir(directory):
        if filename.endswith('.xlsx'):
            # Tạo đường dẫn đầy đủ cho file
            filepath = os.path.join(directory, filename)
            # Đọc file xlsx
            df = pd.read_excel(filepath)
            # Tạo tên file mới cho file CSV
            new_filename = filename.replace('.xlsx', '.csv')
            new_filepath = os.path.join(directory, new_filename)
            # Lưu dataframe dưới dạng file CSV
            df.to_csv(new_filepath, index=False)
            print(f"Đã chuyển đổi: {filepath} sang {new_filepath}")

# Thay thế '/path/to/directory' bằng đường dẫn thực tế của thư mục chứa các file xlsx
directory_path = 'D:/HoiThao/dataxls'
xlsx_to_csv(directory_path)
