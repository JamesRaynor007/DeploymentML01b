import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = FastAPI()

# Cargar los datos de las reseñas
DATA_PATH = 'data.csv'  # Cambia 'data.csv' por el nombre correcto del archivo
Resenias = pd.read_csv(DATA_PATH)

# Convertir los títulos a minúsculas para optimizar las búsquedas
Resenias['title_lower'] = Resenias['title'].str.lower()

# Precomputar la matriz de similitud
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(Resenias['TokensLista'])
cosine_sim = cosine_similarity(count_matrix)

def get_base_url(request: Request):
    scheme = request.url.scheme
    host = request.url.hostname
    port = request.url.port
    return f"{scheme}://{host}:{port}" if port else f"{scheme}://{host}"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return f"""
    <h1>Bienvenido a la API de Recomendación de Películas</h1>
    <p>Esta API te permite obtener recomendaciones de películas basadas en un título dado.</p>
    <p>Ejemplo de uso:</p>
    <p>"url_example": "{get_base_url(request)}/recomendar?title=Inception"</p>
    """

def recomendar_peliculas(data, titulo, num_recomendaciones=5):
    pelicula = data[data['title_lower'] == titulo.lower()]

    if pelicula.empty:
        return f"No se encontró la película: {titulo}"

    voto_original = pelicula['vote_average'].values[0]
    generos = pelicula.iloc[0][4:24]
    generos_compartidos = generos[generos == 1].index.tolist()

    # Filtrar recomendaciones de géneros
    recomendaciones_generos = data[data[generos_compartidos].sum(axis=1) > 0]

    idx = pelicula.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    similar_indices = [i[0] for i in sim_scores if i[0] != idx]

    recomendaciones_similares = data.iloc[similar_indices]
    recomendaciones_finales = pd.concat([recomendaciones_generos, recomendaciones_similares]).drop_duplicates()
    recomendaciones_finales = recomendaciones_finales[recomendaciones_finales['title_lower'] != titulo.lower()]
    recomendaciones_finales = recomendaciones_finales[
        (recomendaciones_finales['vote_average'] >= (voto_original - 1)) &
        (recomendaciones_finales['vote_average'] <= voto_original)
    ]

    recomendaciones_finales['similarity'] = recomendaciones_finales.index.map(lambda x: cosine_sim[idx][x])
    recomendaciones_finales = recomendaciones_finales.sort_values(by=['similarity', 'vote_average'], ascending=False)
    return recomendaciones_finales[['title', 'vote_average']].head(num_recomendaciones).sort_values(by=["vote_average"], ascending=False)

@app.get("/recomendar")
async def recomendar(title: str, num_recomendaciones: int = 5):
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="El archivo de datos no fue encontrado.")

    try:
        recomendaciones = recomendar_peliculas(Resenias, title, num_recomendaciones)

        if isinstance(recomendaciones, str):  # Verifica si se devolvió un mensaje de error
            raise HTTPException(status_code=404, detail=recomendaciones)

        return recomendaciones.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
