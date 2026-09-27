# Resumen de la sesión: Perceptrón, MLP y Word2Vec

## 1. Clasificación binaria con un perceptrón

El dataset Wine tiene 13 variables numéricas por muestra. Para la versión binaria se conservaron solo las clases `0` y `1`.

Un perceptrón binario puede expresarse como:

\[
z = w_1x_1 + w_2x_2 + \dots + w_{13}x_{13} + b
\]

La capa `Linear(13, 1)` aprende los 13 pesos \(w\) y el sesgo \(b\). Su salida es un único **logit** \(z\): un score crudo, no una probabilidad.

Luego, la sigmoide lo transforma en la probabilidad de la clase 1:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Si la probabilidad es mayor que `0.5`, se predice clase `1`; de lo contrario, clase `0`.

### Puntos clave

- `input_size` es la cantidad de variables de entrada (13).
- El segundo parámetro de `nn.Linear(input_size, 1)` es la cantidad de salidas: una probabilidad para la clase 1.
- La probabilidad de la clase 0 queda implícita: `P(clase 0) = 1 - P(clase 1)`.
- `model(X)` ejecuta el método `forward(X)` del modelo; no construye otro modelo.
- En `forward`, `x` representa un lote de muestras: su forma inicial es `(n_muestras, 13)`.

### Trade-off

La sigmoide es no lineal, pero un modelo con una sola capa lineal y una salida sigmoidal mantiene una frontera de decisión lineal: separa según si \(z > 0\). Es simple, rápido e interpretable, pero no puede aprender fronteras complejas.

---

## 2. MLP: perceptrón multicapa

Una red multicapa agrega capas ocultas, por ejemplo:

```text
13 variables → Linear(13, 16) → ReLU → Linear(16, 8) → ReLU → Linear(8, 1) → Sigmoid
```

Cada capa debe recibir la cantidad de features producida por la capa anterior. Por eso no se reutiliza una capa `Linear(13, 1)` varias veces: después de aplicarla, la salida tiene una sola columna y ya no coincide con 13 entradas.

`ReLU` se usa normalmente en capas ocultas:

\[
\operatorname{ReLU}(x) = \max(0, x)
\]

### Trade-off

- Más capas y neuronas permiten aprender relaciones no lineales más ricas.
- También implican más parámetros, más tiempo de entrenamiento y mayor riesgo de sobreajuste.
- Una accuracy de test algo menor que la de train es esperable; una diferencia grande puede señalar sobreajuste o que el modelo no está aprendiendo bien.

---

## 3. Clasificación multiclase

Para las tres clases de Wine (`0`, `1`, `2`), la capa final devuelve tres logits:

```text
Linear(..., 3) → [logit_clase_0, logit_clase_1, logit_clase_2]
```

Ejemplo:

```text
[1.2, -0.3, 2.5]
```

Los logits son scores sin normalizar: pueden ser negativos, mayores a uno y no suman uno.

### Softmax

Softmax convierte los logits en probabilidades:

\[
\operatorname{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

Después de Softmax, cada valor está entre 0 y 1, y todas las probabilidades suman 1.

```text
logits:          [1.2, -0.3, 2.5]
probabilidades:  [0.20, 0.04, 0.76]
```

La predicción se obtiene con `argmax`: el índice de la salida mayor. En el ejemplo, se predice clase `2`.

### Pérdidas

| Problema | Salida del modelo | Pérdida | Etiquetas |
|---|---|---|---|
| Binario | 1 logit + Sigmoid | `BCELoss` | `float`, forma `(N, 1)` |
| Multiclase | C logits, sin Softmax explícito | `CrossEntropyLoss` | `long`, forma `(N,)` |

`CrossEntropyLoss` incorpora internamente Softmax de una manera numéricamente estable; por eso el modelo multiclase debe devolver logits, sin agregar Softmax en `forward`.

Como alternativa binaria más estable, se puede devolver un logit sin Sigmoid y usar `BCEWithLogitsLoss`, que aplica internamente la sigmoide.

---

## 4. Word2Vec

Word2Vec aprende un vector numérico (*embedding*) por palabra a partir de los contextos en los que aparece.

```text
"great" → vector de 100 números
```

La idea central es: palabras que aparecen en contextos similares tienden a tener embeddings cercanos.

### Tokenización simple

El tokenizador transforma cada review en una lista de palabras:

```python
re.compile(r"\b\w+\b", re.UNICODE)
```

- `\b`: límite de palabra.
- `\w+`: una o más letras, números o guiones bajos.
- `lower()`: unifica mayúsculas y minúsculas.
- La puntuación se elimina; por ejemplo, `don't` se divide en `don` y `t`.
- `tqdm` solo muestra una barra de progreso; no cambia los datos.

### Hiperparámetros usados

| Hiperparámetro | Función | Trade-off |
|---|---|---|
| `vector_size=100` | Dimensión de cada embedding | Más dimensiones pueden representar más matices, pero usan más memoria y tardan más en entrenar. |
| `window=5` | Palabras a cada lado consideradas contexto | Una ventana chica captura relaciones locales; una grande incorpora contexto más amplio. |
| `sg=1` | Usa Skip-gram | Skip-gram suele funcionar bien con palabras menos frecuentes, pero puede ser más lento que CBOW. |
| `negative=10` | Cantidad de ejemplos negativos por ejemplo real | Más negativos puede mejorar la representación, pero aumenta el tiempo de entrenamiento. |
| `epochs=5` | Cantidad de pasadas por el corpus | Más épocas pueden mejorar el aprendizaje hasta cierto punto; demasiadas pueden sobreajustar al corpus. |

### Negative sampling

No significa "palabras negativas" ni sentimiento negativo. Para un par real de contexto, por ejemplo `("great", "movie")`, el modelo también toma palabras aleatorias como ejemplos de pares no asociados. Con `negative=10`, usa 10 pares negativos por cada par real. Así evita comparar contra todo el vocabulario y entrena más rápido.

---

## 5. Similitud coseno y relaciones entre palabras

La similitud coseno compara la dirección de dos embeddings:

\[
\operatorname{cos}(A, B) = \frac{A \cdot B}{\|A\|\|B\|}
\]

| Valor | Interpretación geométrica | Interpretación habitual |
|---|---|---|
| `1` | Misma dirección | Máxima similitud contextual |
| `0` | Vectores perpendiculares | Sin relación direccional clara |
| `-1` | Direcciones opuestas | Contextos opuestos, aunque no siempre equivale a antónimos |

La función `w2v.wv.similarity(...)` devuelve **similitud** coseno, no distancia. Una distancia coseno habitual es `1 - similitud`.

### `most_similar`: positive y negative

En `most_similar`, `positive` y `negative` indican operaciones con vectores, no sentimiento:

```text
positive → suma el embedding
negative → resta el embedding
```

Por ejemplo:

```text
king + woman - man
```

construye un vector de consulta y busca las palabras más cercanas a él. `positive=["great"]` simplemente busca palabras cercanas al vector de `great`.

---

## 6. Vector de una review y clasificación de sentimiento

Para clasificar una review completa, se promedian los embeddings de sus palabras:

```text
review → embeddings de sus palabras → promedio → vector de 100 dimensiones
```

Esto permite usar un clasificador tradicional, como regresión logística, sobre vectores numéricos.

### Trade-off del promedio de embeddings

- Es una representación simple, rápida y de tamaño fijo.
- Pierde el orden de las palabras y matices importantes: `"not good"` puede quedar cerca de `"good"`.
- No considera directamente negaciones, estructura gramatical ni la importancia diferencial de cada palabra.

---

## 7. Visualización con t-SNE

t-SNE reduce embeddings de muchas dimensiones (por ejemplo, 100) a dos dimensiones para dibujarlos.

```text
embedding de 100 dimensiones → t-SNE → punto 2D
```

Permite observar vecindades locales: palabras que aparecen cercanas en el gráfico suelen ser similares en el espacio original.

### Trade-off

- Es útil para explorar patrones y grupos locales de palabras.
- No preserva fielmente todas las distancias globales: que dos grupos estén muy separados en el gráfico no prueba que tengan esa separación exacta en el espacio original.
- El resultado puede variar según los parámetros y la semilla.

---

## 8. Word2Vec por sentimiento

Entrenar un Word2Vec solo con reviews positivas y otro solo con reviews negativas permite comparar cómo cambia el contexto de una palabra según el tipo de texto.

```text
reviews con label 1 → Word2Vec positivo
reviews con label 0 → Word2Vec negativo
```

Se pueden contrastar similitudes como `good`–`great`, `bad`–`boring` o `good`–`bad` en ambos modelos.

### Interpretación y cuidado

Las diferencias muestran asociaciones aprendidas dentro de cada subconjunto de reviews. No representan una verdad universal sobre las palabras: dependen del tamaño del corpus, la frecuencia de cada término y los contextos disponibles en cada grupo.
