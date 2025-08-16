# SAM2 - CI/CD Repository

Este repositorio contiene el código fuente de Segment Anything 2 (SAM2) con pipeline de CI/CD configurado.

## 🚀 **Configuración del Repositorio**

### **Archivos Excluidos del Repositorio**
- ✅ **Modelos pre-entrenados** (`checkpoints/*.pt`) - Descargar manualmente
- ✅ **Videos de ejemplo** (`demo/data/gallery/*.mp4`) - Archivos grandes
- ✅ **Archivos de salida** (`output/`) - Generados durante ejecución
- ✅ **Archivos personales** (`test1.py`) - Configuraciones específicas del usuario

### **Descargar Modelos Pre-entrenados**
```bash
cd checkpoints
chmod +x download_ckpts.sh
./download_ckpts.sh
```

## 🔧 **Pipeline CI/CD**

### **GitHub Actions Workflow**
El repositorio incluye un pipeline completo en `.github/workflows/ci-cd-pipeline.yml`:

1. **Testing**: Pruebas en múltiples versiones de Python (3.8, 3.9, 3.10)
2. **Building**: Construcción de imágenes Docker
3. **Deployment**: Despliegue automático a staging
4. **Performance Testing**: Pruebas de rendimiento y precisión

### **Configuración Requerida**

#### **Secrets de GitHub**
Configurar en `Settings > Secrets and variables > Actions`:

```bash
DOCKERHUB_USERNAME=tu_usuario_dockerhub
DOCKERHUB_TOKEN=tu_token_dockerhub
```

#### **Variables de Entorno**
```bash
# Backend
MODEL_SIZE=base_plus
FFMPEG_NUM_THREADS=1
MAX_UPLOAD_VIDEO_DURATION=10

# Frontend
VITE_API_URL=http://localhost:7263
```

## 🐳 **Docker**

### **Construir Imágenes**
```bash
# Backend
docker build -f backend.Dockerfile -t sam2/backend .

# Frontend
cd demo/frontend
docker build -f frontend.Dockerfile -t sam2/frontend .
```

### **Ejecutar con Docker Compose**
```bash
docker-compose up -d
```

## 🧪 **Testing**

### **Pruebas Unitarias**
```bash
pytest tests/ -v --cov=sam2 --cov-report=xml --cov-report=html
```

### **Pruebas de Modelo**
```bash
python -c "from sam2.build_sam import build_sam2_video_predictor; print('Model loading test passed')"
```

## 📊 **Monitoreo**

### **Coverage Reports**
- XML: `coverage.xml` (para CI/CD)
- HTML: `htmlcov/` (para revisión local)

### **Performance Metrics**
- Modelo de precisión
- Tiempo de inferencia
- Uso de memoria GPU

## 🚨 **Consideraciones de Seguridad**

### **Archivos Verificados**
- ✅ No hay claves API hardcodeadas
- ✅ No hay credenciales en el código
- ✅ Variables de entorno configuradas correctamente
- ✅ Rutas relativas en lugar de absolutas

### **Variables de Entorno**
Todas las configuraciones sensibles usan variables de entorno:
```python
MODEL_SIZE = os.getenv("MODEL_SIZE", "base_plus")
API_URL = os.getenv("API_URL", "http://localhost:7263")
```

## 🔄 **Workflow de Desarrollo**

1. **Fork** del repositorio
2. **Branch** para nueva funcionalidad
3. **Commit** con mensajes descriptivos
4. **Push** y **Pull Request**
5. **CI/CD** automático en GitHub Actions
6. **Review** y **Merge**

## 📝 **Notas Importantes**

- Los modelos pre-entrenados NO están incluidos (archivos muy grandes)
- Los videos de ejemplo NO están incluidos (archivos multimedia)
- El pipeline CI/CD está configurado para ejecutarse en cada PR
- Las imágenes Docker se construyen automáticamente en el main branch

## 🆘 **Soporte**

Para problemas con CI/CD:
1. Revisar logs en GitHub Actions
2. Verificar secrets configurados
3. Comprobar compatibilidad de versiones
4. Revisar archivos de configuración 