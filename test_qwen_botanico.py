import sys
import transformers.tokenization_utils_base

# PARCHE DE COMPATIBILIDAD IAG-ARQUI:
# Mapear 'vocab' para evitar AttributeError en tokenizadores lentos (Qwen2)
if not hasattr(transformers.tokenization_utils_base.PreTrainedTokenizerBase, "vocab"):
    transformers.tokenization_utils_base.PreTrainedTokenizerBase.vocab = property(
        lambda self: self.get_vocab()
    )

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
# mlx_vlm.utils.load_config is required internally sometimes, but load returns the processor
from mlx_vlm.utils import load_config

def test_qwen_botanico(image_path: str):
    print("========================================")
    print(" INICIANDO SECUENCIA QWEN2.5-VL (Apple MLX) ")
    print("========================================")
    
    # Este model_path descargará el modelo de 4-bit automáticamente si no existe (~4.5GB)
    model_path = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    
    print(f"[*] Cargando Modelo Fundacional: {model_path} ...")
    print(f"[!] ADVERTENCIA: Puede tomar varios minutos si es la primera vez (Descarga de Pesos).")
    
    # Carga optimizada: El cargador de MLX espera el DIRECTORIO del adaptador
    adapter_path = "." # Donde residen adapters.safetensors y adapter_config.json
    print(f"[*] Inyectando Sabiduría Experta desde directorio: {adapter_path}")
    
    # El cargador de mlx_vlm gestiona internamente la fusión del adaptador
    model, processor = load(model_path, adapter_path=adapter_path)
    config = load_config(model_path)
    print("[*] Pesos Neuronales y Adaptador LoRA montados en la RAM Unificada.")

    prompt_texto = (
        "Eres el Científico Agrónomo Experto de IAG-ARQUI. Analiza esta imagen con "
        "extremo rigor botánico y detalla clínicamente el estado fitosanitario de la "
        "planta o cultivo de caña de azúcar mostrado. Busca cualquier tipo de anomalía, "
        "enfermedad fúngica, bacteriana, viral o carencia nutricional. Si identificas patologías, "
        "nombra la enfermedad específica y sus síntomas obvios. Si la planta parece sana y libre de "
        "riesgos, indícalo claramente. Responde en un párrafo breve y concluyente."
    )

    print(f"[*] Inyectando el Tensor Fotográfico: {image_path}")
    # Formato de chat universal de mlx-vlm
    prompt = apply_chat_template(
        processor, 
        config, 
        prompt_texto, 
        num_images=1
    )

    print("[*] Pensando... (Inferencia sobre M4 en progreso)")
    # verbose=True imprimirá los tokens uno por uno como si estuviera tipeando!
    output = generate(
        model, 
        processor, 
        prompt, 
        image=[image_path], 
        max_tokens=600, 
        temperature=0.2, # Baja temperatura para que no invente (alucinación mínima)
        verbose=True
    )
    
    print("\n========================================")
    print(" DIAGNÓSTICO CLÍNICO COMPLETADO ")
    print("========================================")

if __name__ == "__main__":
    # Test images that the user uploaded previously
    # Photo 1 (Macro Rust): The user uploaded it, let's assume it's in the root or ask them for the path.
    # We will use test2.jpg (Macro Rust) if it exists, or just allow parameter passing.
    if len(sys.argv) > 1:
        target_img = sys.argv[1]
    else:
        # Pangea test images usually exist in the root or VisorCana uploads.
        # Let's use a dummy path to remind the user to pass the real one.
        target_img = "dataset/images/train/test_leaf.jpg" # Ajustar según existan
        print("Uso: python test_qwen_botanico.py <ruta_a_la_foto.jpg>")
    
    try:
        test_qwen_botanico(target_img)
    except FileNotFoundError:
        print(f"[Error] La foto {target_img} no existe. Por favor, especifica una foto real.")
