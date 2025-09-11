import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.cm as cm

# Eventos ordenados cronológicamente (fecha, etiqueta)
events = [
    ("1538-01-01", "Fundacion Bogota"),
    ("1538-01-01", "Capilla humilladero"),
    ("1553-01-01", "Catedral Primada"),
    ("1557-01-01", "Iglesia San Francisco"),
    ("1633-01-01", "Iglesia Egipto"),
    ("1636-01-01", "Iglesia la candelaria"),
    ("1640-01-01", "Ermita Monserrate"),
    ("1656-01-01", "Ermita Guadalupe")
]

# Parámetros editables
point_size = 120         # Tamaño de los puntos
line_width = 3           # Grosor de las líneas
font_size_legend = 12    # Tamaño fuente leyenda

# Convertir fechas a datetime
dates = [datetime.strptime(date, "%Y-%m-%d") for date, _ in events]
labels = [label for _, label in events]

# Colores de la paleta terrain, distribuidos uniformemente
cmap = plt.get_cmap('tab10')  # Colores más contrastantes para fondo blanco
colors = [cmap(i / (len(events) - 1)) for i in range(len(events))]

fig, ax = plt.subplots(figsize=(12, 3))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Dibujar líneas coloreadas antes de cada punto
for i in range(1, len(dates)):
    ax.hlines(1, dates[i-1], dates[i], color=colors[i], linewidth=line_width, zorder=2, alpha=0.8)

# Dibujar puntos con label para la leyenda
for date, label, color in zip(dates, labels, colors):
    ax.scatter(date, 1, color=color, s=point_size, label=label, zorder=3, edgecolors='black', linewidth=0.5)

# Configurar eje x para mostrar solo años con ticks cada 5 años
ax.xaxis.set_major_locator(mdates.YearLocator(base=5))  # Tick cada 5 años
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(color='black', rotation=45)

# Limpiar eje y y bordes
ax.yaxis.set_visible(False)
for spine in ax.spines.values():
    spine.set_visible(False)

# Crear leyenda lateral con texto blanco y fondo negro
leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                fontsize=font_size_legend, frameon=True, labelspacing=1.2,
                title="Eventos", title_fontsize=font_size_legend+2)

# Cambiar color del texto de la leyenda a blanco
for text in leg.get_texts():
    text.set_color('black')

# Cambiar color del título de la leyenda a blanco
leg.get_title().set_color('black')

# Cambiar fondo y borde de la leyenda para que contraste con el fondo negro
leg.get_frame().set_facecolor('white')
leg.get_frame().set_edgecolor('black')
leg.get_frame().set_linewidth(0.8)

plt.tight_layout(rect=[0, 0, 0.85, 1])  # Deja espacio para la leyenda a la derecha
plt.savefig("/home/usuaryo/Pictures/timeline_events.png", bbox_inches='tight', dpi=500)
plt.show()
