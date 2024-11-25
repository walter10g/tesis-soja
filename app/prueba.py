import os

# Ruta base de las carpetas
base_path = "datasets/colors/images"

# Obtener todas las carpetas dentro del directorio base
categories = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]

# Renombrar archivos en cada carpeta
for category in categories:
    category_path = os.path.join(base_path, category)
    files = os.listdir(category_path)
    
    # Ordenar los archivos para renombrarlos secuencialmente
    files.sort()
    
    for idx, file in enumerate(files):
        old_file_path = os.path.join(category_path, file)
        if os.path.isfile(old_file_path):  # Ignorar subdirectorios
            new_file_name = f"{category}_{idx + 1}.jpg"
            new_file_path = os.path.join(category_path, new_file_name)
            
            # Renombrar archivo
            os.rename(old_file_path, new_file_path)
            print(f"Renombrado: {old_file_path} -> {new_file_path}")

print("Renombrado completo.")
