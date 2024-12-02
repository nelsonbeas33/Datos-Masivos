from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StructType, StructField, ArrayType, StringType
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from pyspark.sql.functions import size, expr, col, lower, regexp_replace, lit, trim, when
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import json

# Iniciar sesión de Spark
spark = SparkSession.builder \
    .appName("Transformer Model with Spark") \
    .getOrCreate()

# Cargar archivos CSV
#products_df = spark.read.option("header", "true").csv("/opt/products_test.csv")

products_df = spark.read.option("header", "true").csv("/opt/products.csv").limit(2000)
# Renombrar las columnas 'product_id' para evitar ambigüedad después del join
products_df = products_df.withColumnRenamed("product_id", "product_id")

products_df.show(10)

# Asegurar que las columnas 'order_product_id' y 'product_id' sean enteros
products_df = products_df.withColumn("product_id", products_df["product_id"].cast(IntegerType()))

# Crear la columna 'processed_product_name'
stopwords = ["with", "and", "in", "of", "the", "for", "a", "an", "free", "mix", "original", "pack"]

# Generar una expresión regular para eliminar stopwords
stopwords_regex = r'\b(?:' + '|'.join(stopwords) + r')\b'

# Procesar el nombre del producto
products_df = products_df.withColumn(
    "processed_product_name",
    trim(
        regexp_replace(
            regexp_replace(
                regexp_replace(
                    regexp_replace(lower(col("product_name")), r'\d+', ''),  # Eliminar números y convertir a minúsculas
                    r'[^a-z\s]', ''                                         # Eliminar caracteres especiales
                ),
                stopwords_regex, ''                                         # Eliminar stopwords
            ),
            r'\s+', ' '                                                    # Reemplazar múltiples espacios por uno solo
        )
    )
)


# Crear el vectorizador TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 1), max_features=1000)

# Ajustar y transformar la lista de productos
processed_names = products_df.select("processed_product_name").rdd.flatMap(lambda x: x).collect()
tfidf_matrix = vectorizer.fit_transform(processed_names)

# Obtener los nombres de las características (palabras)
words = vectorizer.get_feature_names_out()

# Sumar los valores de TF-IDF para cada palabra en todos los productos
sum_tfidf = np.asarray(tfidf_matrix.sum(axis=0)).flatten()

# Crear un diccionario de palabras con su respectiva suma de TF-IDF
word_tfidf = {words[i]: sum_tfidf[i] for i in range(len(words))}

# Ordenar las palabras por la suma de sus valores de TF-IDF, de mayor a menor
sorted_words = sorted(word_tfidf.items(), key=lambda item: item[1], reverse=True)

# Obtener las n palabras más importantes
top_n_words_rank = sorted_words[:30]

# Imprimir las palabras más importantes
#print("Top n palabras más importantes:")
#for word, score in top_n_words_rank:
    #print(f"{word}: {score}")

# Ahora, aplicamos el vectorizador
top_n_words = [word for word, score in top_n_words_rank]
vectorizer = TfidfVectorizer(vocabulary=top_n_words)

# Ajustar el vectorizador con los nombres de los productos
vectorizer.fit(processed_names)

# Crear una matriz de características de los productos
X = vectorizer.transform(processed_names)

# Convertir a una matriz densa para ver las representaciones
dense_matrix = X.toarray()


# Calcular la matriz de similitud coseno entre los productos, en este caos no se usa ya que kmeans no lo necesita, pero DBSCAN si
#similarity_matrix = cosine_similarity(dense_matrix)

# Número de clusters deseados (esto depende de tu caso, puedes probar diferentes valores)
n_clusters = 20

# Aplicar KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(dense_matrix)

# Agrupar los productos por clúster
clustered_products = {}
for product, cluster_id in zip(processed_names, clusters):
    if cluster_id not in clustered_products:
        clustered_products[cluster_id] = []
    clustered_products[cluster_id].append(product)

#Imprimir los productos agrupados por clúster
#for cluster_id, products in clustered_products.items():
    #print(f"\nCluster {cluster_id}:")
    #for product in products:
        #print(f"  - {product}") 

# Asegúrate de que mapping_list contenga solo enteros estándar (int)
mapping_list = []
for cluster_id, products in clustered_products.items():
    for product in products:
        # Asegúrate de que el virtual_product_id sea un entero estándar
        mapping_list.append((product, int(cluster_id + 1)))  # Convertimos explícitamente a int

# Especificamos el esquema explícitamente
schema = StructType([
    StructField("processed_product_name", StringType(), True),
    StructField("virtual_product_id", IntegerType(), True)  # Especificamos que esta columna es de tipo entero
])

# Crear el DataFrame de mapeo con el esquema especificado
mapping_cluster_df = spark.createDataFrame(mapping_list, schema)

# Ahora realizar el join con el DataFrame original
products_df = products_df.join(mapping_cluster_df, on="processed_product_name", how="left")

# Mostrar los resultados
products_df.show(10)

order_products_df = spark.read.option("header", "true").csv("/opt/order_products__prior.csv").limit(100000)
order_products_df_test = spark.read.option("header", "true").csv("/opt/order_products__train.csv")

order_products_df = order_products_df.withColumnRenamed("product_id", "order_product_id")
order_products_df_test = order_products_df_test.withColumnRenamed("product_id", "order_product_id")
order_products_df.show(10)

print("total de productos asignados a alguna orden:", order_products_df.count())
print("total de productos:", products_df.count())

# Asegurar que las columnas 'order_product_id' y 'product_id' sean enteros
order_products_df = order_products_df.withColumn("order_product_id", order_products_df["order_product_id"].cast(IntegerType()))
order_products_df_test = order_products_df_test.withColumn("order_product_id", order_products_df_test["order_product_id"].cast(IntegerType()))

# Realizar un join entre los productos y las órdenes
joined_df = order_products_df.join(products_df, order_products_df.order_product_id == products_df.product_id)
joined_df_df_test = order_products_df_test.join(products_df, order_products_df_test.order_product_id == products_df.product_id)

joined_df.show(10)

# Agrupar por 'order_id' y crear las listas de productos, pasillos y departamentos
order_grouped_df = joined_df.groupBy("order_id").agg(
    F.collect_list("virtual_product_id").alias("virtual_product_id"),
    F.collect_list("product_id").alias("product_id"),
    F.collect_list("aisle_id").alias("aisle_ids"),
    F.collect_list("department_id").alias("department_ids"),
    F.collect_list("processed_product_name").alias("processed_product_name")
)

# Agrupar por 'order_id' y crear las listas de productos, pasillos y departamentos
order_grouped_df_test = joined_df_df_test.groupBy("order_id").agg(
    F.collect_list("virtual_product_id").alias("virtual_product_id"),
    F.collect_list("product_id").alias("product_id"),
    F.collect_list("aisle_id").alias("aisle_ids"),
    F.collect_list("department_id").alias("department_ids"),
    F.collect_list("processed_product_name").alias("processed_product_name")
)

order_grouped_df = order_grouped_df \
    .withColumn("aisle_ids", F.expr("transform(aisle_ids, x -> cast(x as int))")) \
    .withColumn("department_ids", F.expr("transform(department_ids, x -> cast(x as int))"))

order_grouped_df_test = order_grouped_df_test \
    .withColumn("aisle_ids", F.expr("transform(aisle_ids, x -> cast(x as int))")) \
    .withColumn("department_ids", F.expr("transform(department_ids, x -> cast(x as int))"))
order_grouped_df.show(10)

filtered_df = order_grouped_df.filter(size("virtual_product_id") > 1)
filtered_df_test = order_grouped_df_test.filter(size("virtual_product_id") > 1)

# Definir la lógica para el número de elementos a eliminar
#filtered_df = filtered_df.withColumn(
    #"remove_count",
    #F.when(size(col("virtual_product_id")) <= 3, 1)  # Si el tamaño es 2 o 3, eliminar 1
    #.otherwise(F.expr("cast(rand() * 2 + 1 as int)"))  # Si es >=4, eliminar entre 1 y 2
#)

filtered_df = filtered_df.withColumn(
    "remove_count", 
    F.lit(1)  # Asigna siempre el valor 1
)

filtered_df_test = filtered_df_test.withColumn(
    "remove_count", 
    F.lit(1)  # Asigna siempre el valor 1
)

# Crear nuevas columnas con los elementos eliminados
filtered_df = filtered_df \
    .withColumn("removed_order_product_ids", expr("slice(virtual_product_id, size(virtual_product_id) - remove_count + 1, remove_count)")) \
    .withColumn("removed_aisle_ids", expr("slice(aisle_ids, size(aisle_ids) - remove_count + 1, remove_count)")) \
    .withColumn("removed_department_ids", expr("slice(department_ids, size(department_ids) - remove_count + 1, remove_count)"))

# Crear nuevas columnas con los elementos eliminados
filtered_df_test = filtered_df_test \
    .withColumn("removed_order_product_ids", expr("slice(virtual_product_id, size(virtual_product_id) - remove_count + 1, remove_count)")) \
    .withColumn("removed_aisle_ids", expr("slice(aisle_ids, size(aisle_ids) - remove_count + 1, remove_count)")) \
    .withColumn("removed_department_ids", expr("slice(department_ids, size(department_ids) - remove_count + 1, remove_count)"))

# Actualizar las columnas originales para conservar los elementos restantes
filtered_df = filtered_df \
    .withColumn("virtual_product_id", expr("slice(virtual_product_id, 1, size(virtual_product_id) - remove_count)")) \
    .withColumn("aisle_ids", expr("slice(aisle_ids, 1, size(aisle_ids) - remove_count)")) \
    .withColumn("department_ids", expr("slice(department_ids, 1, size(department_ids) - remove_count)"))

# Actualizar las columnas originales para conservar los elementos restantes
filtered_df_test = filtered_df_test \
    .withColumn("virtual_product_id", expr("slice(virtual_product_id, 1, size(virtual_product_id) - remove_count)")) \
    .withColumn("aisle_ids", expr("slice(aisle_ids, 1, size(aisle_ids) - remove_count)")) \
    .withColumn("department_ids", expr("slice(department_ids, 1, size(department_ids) - remove_count)"))


filtered_df.show(20)

def create_binary_encoding(removed_product_ids, vocab_size):
    # Crear una lista de codificaciones para cada producto retirado
    encodings = []
    
    # Para cada producto retirado, generar su propia codificación binaria
    for product_id in removed_product_ids:
        encoding = torch.zeros(vocab_size)
        encoding[product_id] = 1  # Marcar el índice correspondiente como 1
        encodings.append(encoding.unsqueeze(0))  # Agregar dimensión extra (1, vocab_size)
    
    # Devolver la lista de codificaciones
    return torch.cat(encodings, dim=0)  # Concatenar las codificaciones a lo largo del eje 0 (batch)




class ProductRetirementModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_prob):
        super(ProductRetirementModel, self).__init__()
        
        # Capa de embedding para convertir los productos en vectores
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM para procesar las secuencias de productos
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Capa fully connected para predecir la probabilidad de cada producto
        self.fc = nn.Linear(hidden_dim, vocab_size)  # Cada producto tiene una probabilidad de ser retirado
        
        # Dropout para regularización
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Paso de embedding
        embedded = self.embedding(x)
        
        # Pasamos las secuencias por el LSTM
        rnn_out, (h, c) = self.rnn(embedded)
        
        # Verificamos la dimensionalidad de rnn_out
        if rnn_out.dim() == 2:
            # Si la secuencia es de longitud 1, debemos agregar una dimensión adicional
            rnn_out = rnn_out.unsqueeze(1)
        
        # Usamos la salida final del LSTM
        out = self.fc(rnn_out[:, -1, :])  # Usar la salida del último paso de la secuencia
        
        # Aplicamos dropout
        out = self.dropout(out)
        
        # Devolvemos las probabilidades de que cada producto esté retirado
        return torch.sigmoid(out)  # Probabilidades entre 0 y 1

# Función para entrenar el modelo
def train_model(model, data, epochs=10, learning_rate=0.001):
    # Optimizer y función de pérdida
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()  # Usamos BCEWithLogitsLoss porque la salida es una probabilidad entre 0 y 1
    
    model.train()  # Establecemos el modelo en modo de entrenamiento

    losses = []
    accuracies = []

    for epoch in range(epochs):
        epoch_loss = 0
        for register in data:            
            ticket_product = torch.tensor(register["virtual_product_id"], dtype=torch.long).unsqueeze(0)  # Hacer el batch de tamaño 1
            removed_product = torch.tensor(register["removed_order_product_ids"], dtype=torch.long)  # Hacer el batch de tamaño 1
            # Pasamos la entrada por el modelo
            optimizer.zero_grad()
            #print("input: ", ticket_product)
            #print(f"Dimensiones input: {ticket_product.shape}")
            outputs = model(ticket_product)

            #print("removed_order_product_ids: ", removed_product)
            #print("outputs: ", outputs)
            #print("removed_product: ", create_binary_encoding(removed_product, vocab_size))

            loss = loss_fn(outputs, create_binary_encoding(removed_product, vocab_size))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/item_number} - ")
        losses.append(epoch_loss/item_number)
        accuracies.append(evaluate_model_accuracy(model, filtered_df_test.collect(), vocab_size))

    print("perdidas")
    for loss in losses:
        print(loss)

    print("accuracies")
    for acc in accuracies:
        print(acc)

def evaluate_model_accuracy(model, data, vocab_size):
    model.eval()  # Establecemos el modelo en modo de evaluación
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # Desactivamos el cálculo de gradientes para la evaluación
        for register in data:
            ticket_product = torch.tensor(register["virtual_product_id"], dtype=torch.long).unsqueeze(0)
            removed_product = torch.tensor(register["removed_order_product_ids"], dtype=torch.long)
            
            # Pasamos la entrada por el modelo
            outputs = model(ticket_product)
            
            # Convertimos las salidas a probabilidades utilizando sigmoid
            predicted_probs = torch.sigmoid(outputs)  # Esto nos da las probabilidades entre 0 y 1
            
            # Convertimos removed_product en formato binario (0 o 1)
            target = create_binary_encoding(removed_product, vocab_size)
            
            # Predicción: tomar el índice con la mayor probabilidad y convertirlo en 1, el resto en 0
            predicted = torch.zeros_like(predicted_probs, dtype=torch.int)  # Inicializar con ceros
            max_index = torch.argmax(predicted_probs)  # Encontrar el índice con la mayor probabilidad
            predicted[0, max_index] = 1  # Marcar como 1 el índice con la mayor probabilidad
            
            # Calculamos las predicciones correctas
            #print("predict: ", predicted)
            #print("target: ", target)
            # Encuentra el índice marcado como 1 en el target (producto que fue retirado)
            target_idx = torch.argmax(target)  # Encuentra el índice con el valor 1 en 'target'

            # Compara si el índice predicho por el modelo coincide con el índice real
            if max_index == target_idx:
                correct_predictions += 1  # Si el índice predicho es correcto, incrementa el contador
                #print("se detecto el producto correcto: ", target_idx, "vs", max_index)

            total_predictions += 1  # Incrementa el número total de predicciones evaluadas

    accuracy = correct_predictions / total_predictions * 100
    print(f"Accuracy: {accuracy:.2f}%", " - correct: ", correct_predictions, " de ", total_predictions)
    return accuracy
    

# Parámetros
vocab_size = products_df.select("virtual_product_id").distinct().count() + 1
hidden_dim = 128  # Dimensión de la capa oculta del LSTM
max_len = 20  # Longitud máxima de la secuencia
epochs=10
learning_rate=0.001
embedding_dim = 256
item_number = filtered_df.count()
item_number_test = filtered_df_test.count()
dropout_prob = 0.3

print("vocab_size", vocab_size)
print("hidden_dim", hidden_dim)
print("max_len", max_len)
print("epochs", epochs)
print("learning_rate", learning_rate)
print("embedding_dim", embedding_dim)
print("ordenes", item_number)
print("ordenes test", item_number_test)

# Crear el modelo
model = ProductRetirementModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, dropout_prob=dropout_prob)

# Si tienes GPU disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Entrenar el modelo
train_model(model, filtered_df.collect(), epochs, learning_rate)

hyperparameters = {
    "vocab_size": vocab_size,
    "embedding_dim": embedding_dim,
    "hidden_dim": hidden_dim,
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": 1,  # Puedes agregar cualquier otro hiperparámetro relevante
    "dropout_prob": 0.3
}

def save_hyperparameters(hyperparameters, filepath="/tmp/hyperparameters.json"):
    with open(filepath, "w") as f:
        json.dump(hyperparameters, f, indent=4)
    print(f"Hiperparámetros guardados en {filepath}")

torch.save(model, "/tmp/product_retirement_full_model.pth")

np.save("/tmp/dense_matrix.npy", dense_matrix)
np.save("/tmp/clusters.npy", clusters)
