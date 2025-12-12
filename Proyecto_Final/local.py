import requests
import json
import argparse
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# ============================================
# PARSEO DE ARGUMENTOS
# ============================================
parser = argparse.ArgumentParser(description="Enviar imagen al Space de Hugging Face y mostrar predicciones.")
parser.add_argument("--image", type=str, required=True, help="Ruta de la imagen")
args = parser.parse_args()

IMAGE_PATH = args.image
API_URL = "https://juannmontoya-billar-detector-v1.hf.space/predict"

print(f"📡 Conectando a: {API_URL}")
print(f"📁 Imagen: {IMAGE_PATH}")

try:
    with open(IMAGE_PATH, "rb") as f:
        # Enviamos la imagen como un archivo form-data
        response = requests.post(API_URL, files={"file": f})

    if response.status_code == 200:
        data = response.json()
        print("\n✅ ¡ÉXITO! Respuesta del servidor:")
        print(json.dumps(data, indent=2))
        
        # --- DIBUJAR RESULTADO ---
        print("\n🎨 Dibujando resultado...")
        img = Image.open(IMAGE_PATH).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Fuente predeterminada de PIL
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except:
            font = ImageFont.load_default()

        for det in data.get('detections', []):
            label = det['label']
            box = det['bbox']
            conf = det['confidence']
            tipo = det['type']

            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if tipo == "zone" else (255, 0, 0)  # Verde/rojo

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, y1 - 15), f"{label} {conf:.2f}", fill=color, font=font)
        
        # Mostrar imagen con matplotlib
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title("Predicciones del Space")
        plt.show()

        # Guardar imagen con detecciones
        output_path = "resultado_detecciones.jpg"
        img.save(output_path)
        print(f"✅ Imagen guardada como {output_path}")
        
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

except Exception as e:
    print(f"❌ Error de conexión: {e}")
    print("Consejo: Verifica que el Space esté 'Running' y que la URL sea la correcta (.hf.space)")
