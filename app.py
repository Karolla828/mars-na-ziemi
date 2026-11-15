from pathlib import Path
import html
import json
import math

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


st.set_page_config(
    page_title="Mars na Ziemi",
    page_icon="🪐",
    layout="wide",
)

DATA = Path(__file__).resolve().parent / "data"

AREAS_FILE = DATA / "obszary_porownanie.csv"
METHOD_FILE = DATA / "metoda_porownania.json"

CATEGORIES = {
    "terrain_score": "Rzeźba terenu",
    "temperature_score": "Warunki termiczne",
    "material_score": "Materiał powierzchni",
    "water_score": "Ślady działania wody",
}


@st.cache_data
def load_data(areas_modified, method_modified):
    areas = pd.read_csv(AREAS_FILE)

    method = json.loads(
        METHOD_FILE.read_text(encoding="utf-8")
    )

    return areas, method


def distance_km(first, second):
    lat1, lon1, lat2, lon2 = map(
        math.radians,
        [
            first["latitude"],
            first["longitude"],
            second["latitude"],
            second["longitude"],
        ],
    )

    value = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin((lon2 - lon1) / 2) ** 2
    )

    return 12742 * math.asin(
        math.sqrt(min(1.0, max(0.0, value)))
    )


def select_places(ranking):
    selected = []

    # Wybieramy do pięciu miejsc osobno dla każdego regionu.
    for _, region in ranking.groupby(
        "region_code",
        sort=False,
    ):
        regional_places = []

        for _, row in region.iterrows():
            candidate = row.to_dict()

            if all(
                distance_km(candidate, previous) >= 12
                for previous in regional_places
            ):
                regional_places.append(candidate)

            if len(regional_places) == 5:
                break

        selected.extend(regional_places)

    return sorted(
        selected,
        key=lambda place: (-place["score"], place["id"]),
    )


st.title("🪐 Mars na Ziemi")

st.write(
    "Porównaj fragment Jezero z terenami Atakamy, "
    "Kotliny Tunkińskiej na Syberii, Kapadocji w Turcji i Hanksville w USA."
)

if not AREAS_FILE.exists() or not METHOD_FILE.exists():
    st.error(
        "W folderze data brakuje obszary_porownanie.csv "
        "lub metoda_porownania.json."
    )
    st.stop()

areas, method = load_data(
    AREAS_FILE.stat().st_mtime_ns,
    METHOD_FILE.stat().st_mtime_ns,
)

st.header("1. Obszar Marsa")

st.selectbox(
    "Obszar odniesienia",
    ["Jezero — zachodnia delta i przyległe dno"],
)

st.caption(
    "Porównywane fragmenty mają około 11,3 × 11,9 km. "
    "Wyszukiwanie obejmuje przygotowane fragmenty "
    "Atakamy, Kotliny Tunkińskiej, Kapadocji i Hanksville."
)

st.header("2. Wybierz kryteria podobieństwa")

with st.form("criteria_two_regions"):
    choices = {}

    defaults = {
        "terrain_score": (True, 70),
        "temperature_score": (False, 30),
        "material_score": (False, 30),
        "water_score": (False, 30),
    }

    for column, label in CATEGORIES.items():
        left, right = st.columns([1, 2])

        checked_default, weight_default = defaults[column]

        checkbox_label = label

        if column == "temperature_score":
            checkbox_label += " — eksperymentalne"
        elif column == "material_score":
            checkbox_label += " — rozpoznana część"
        elif column == "water_score":
            checkbox_label += " — Atakama i Hanksville"

        with left:
            checked = st.checkbox(
                checkbox_label,
                value=checked_default,
                key=f"check_{column}",
            )

        with right:
            weight = st.slider(
                f"Znaczenie: {label.lower()}",
                min_value=0,
                max_value=100,
                value=weight_default,
                step=5,
                key=f"weight_{column}",
            )

        choices[column] = {
            "checked": checked,
            "weight": weight,
        }

    submitted = st.form_submit_button(
        "Znajdź podobne miejsca",
        type="primary",
    )

st.caption(
    "Wagi określają znaczenie kategorii względem siebie. "
    "Odznaczenie lub waga 0 pomija kategorię. "
    "Po zmianie ustawień ponownie kliknij przycisk."
)

if submitted:
    st.session_state.pop("two_region_results", None)

    active = {
        column: settings["weight"]
        for column, settings in choices.items()
        if settings["checked"] and settings["weight"] > 0
    }

    if not active:
        st.warning(
            "Zaznacz przynajmniej jedną kategorię "
            "z wagą większą od zera."
        )
    else:
        candidates = areas.replace(
            [np.inf, -np.inf],
            np.nan,
        ).copy()

        required = [
            "latitude",
            "longitude",
            *active.keys(),
        ]

        candidates = candidates.dropna(
            subset=required
        ).copy()

        for column in active:
            candidates = candidates[
                candidates[column].between(0, 100)
            ].copy()

        if candidates.empty:
            st.warning(
                "Nie ma obszarów z kompletem danych "
                "dla wybranych kategorii."
            )
        else:
            columns = list(active)
            weights = np.array(
                list(active.values()),
                dtype=float,
            )

            candidates["score"] = (
                candidates[columns].to_numpy(dtype=float)
                @ weights
                / weights.sum()
            )

            ranking = candidates.sort_values(
                ["score", "id"],
                ascending=[False, True],
            )

            st.session_state["two_region_results"] = {
                "places": select_places(ranking),
                "active": active,
                "candidate_count": len(candidates),
                "region_counts": (
                    candidates.groupby("region")
                    .size()
                    .to_dict()
                ),
            }

st.header("3. Mapa podobnych miejsc")

results = st.session_state.get("two_region_results")

map_view = folium.Map(
    location=[15, 20],
    zoom_start=2,
    tiles="OpenStreetMap",
    control_scale=True,
)

if results:
    active = results["active"]
    places = results["places"]

    if "water_score" in active:
        st.info(
            "Wyszukiwanie obejmuje Atakamę i Hanksville. "
            "Syberia i Turcja nie mają przygotowanych danych śladów wody. "
            "Odznacz tę kategorię, aby porównać oba regiony."
        )

    counts_text = " · ".join(
        f"{region}: {count}"
        for region, count in results["region_counts"].items()
    )

    st.caption(
        "Liczba porównywanych fragmentów — " + counts_text
    )

    for position, place in enumerate(places, start=1):
        lines = []

        for column in active:
            label = CATEGORIES[column]

            if column == "temperature_score":
                label += "*"

            lines.append(
                f"{label}: "
                f"<b>{place[column]:.1f}/100</b>"
            )

        notes = []

        if "temperature_score" in active:
            notes.append(
                "* Temperatura: porównanie eksperymentalne."
            )

            if place["region_code"] == "syberia":
                notes.append(
                    "Temperatura Syberii może dotyczyć śniegu."
                )

        if "material_score" in active:
            notes.append(
                "Materiał rozpoznany: "
                f"Mars {place['mars_material_coverage']:.1f}%, "
                f"Ziemia {place['earth_material_coverage']:.1f}%."
            )

        if "water_score" in active:
            notes.append(
                "Ślady wody: porównanie udziału "
                "zmapowanych osadów."
            )

        notes.append(
            "Wynik dotyczy wybranych cech obszaru."
        )

        region_name = html.escape(str(place["region"]))
        scores_html = "<br>".join(lines)
        notes_html = "<br>".join(notes)

        content = f"""
        <div style="
            font-family:sans-serif;
            width:245px;
            max-width:100%;
            white-space:normal;
            line-height:1.45;
            font-size:13px;
        ">
            <b>{region_name}</b><br>
            Obszar {int(place['id'])}

            <div style="margin:7px 0 10px;">
                Dopasowanie do Jezero:
                <b>{place['score']:.1f}/100</b>
            </div>

            {scores_html}

            <div style="
                margin-top:10px;
                font-size:11px;
                line-height:1.35;
            ">
                {notes_html}
            </div>
        </div>
        """

        folium.Marker(
            location=[
                place["latitude"],
                place["longitude"],
            ],
            tooltip=folium.Tooltip(
                content,
                sticky=False,
            ),
            popup=folium.Popup(
                content,
                max_width=350,
            ),
            icon=folium.Icon(
                color="red" if position == 1 else "blue",
                icon="info-sign",
            ),
        ).add_to(map_view)

    if places:
        map_view.fit_bounds(
            [
                [place["latitude"], place["longitude"]]
                for place in places
            ],
            padding=(35, 35),
            max_zoom=9,
        )

    st.caption(
        "Do pięciu najwyżej ocenionych miejsc z każdego regionu. "
        "Środki wybranych miejsc w regionie dzieli co najmniej 12 km. "
        "Czerwona pinezka ma najwyższy wynik spośród pokazanych. "
        "Najedź na pinezkę lub ją kliknij."
    )
else:
    st.caption(
        "Wybierz kryteria i kliknij „Znajdź podobne miejsca”."
    )

st_folium(
    map_view,
    height=600,
    use_container_width=True,
    returned_objects=[],
    key="two_region_map",
)

with st.expander("Źródła danych i sposób liczenia"):
    st.write(
        "Obszary Ziemi: przygotowane fragmenty Atakamy, "
        "Kotliny Tunkińskiej (51–52°N, 102–103°E) "
        "Kapadocji (38–39°N, 34–35°E) "
        "i okolic Hanksville (38–39°N, 111–110°W). "
        "Wzorzec Marsa: wybrany fragment Jezero."
    )

    st.write(
        "Rzeźba: model CTX/USGS z danych NASA MRO dla Marsa "
        "oraz Terrain Tiles dla Ziemi. Siatka analizy 100 m. "
        "Wspólne skale różnic obliczono ze wszystkich czterech regionów Ziemi. "
        "Nachylenie, przewyższenia i nierówność mają "
        "wagi wewnętrzne 40/30/30."
    )

    st.write(
        "Temperatura: MEDA/TIRS, sol 30 Marsa, oraz "
        "NASA POWER/MERRA-2, marzec 2021 Ziemi. "
        "Porównujemy 24 mediany godzinowe. "
        "Indeks = 100 / (1 + średnia różnica / 20°C). "
        "Parametr 20°C jest umowny. Marsjańska temperatura "
        "jasnościowa i ziemska temperatura z reanalizy "
        "nie są identycznym rodzajem pomiaru. "
        "Flaga położenia koła łazika jest nieznana. "
        "Śnieg i roślinność ograniczają interpretację Syberii."
    )

    st.write(
        "Materiał: mapa Jezero Sun i Stack (USGS, 2020) "
        "oraz GLiM dla Ziemi. Porównanie proporcji szerokich "
        "klas wyłącznie w rozpoznanej części powierzchni."
    )

    st.write(
        "Ślady wody: dostępne dla Atakamy i Hanksville. "
        "Źródła: mapa Jezero, Geología Base Chile oraz USGS SGMC dla Utah. "
        "Porównanie udziału zmapowanych osadów rzecznych, "
        "aluwialnych i jeziornych. "
        "Indeks = 100 / (1 + różnica udziałów / 0,20). "
        "Wymagane pokrycie mapą co najmniej 95%."
    )

    st.write(
        "Ocena łączna jest średnią ważoną aktywnych kategorii. "
        "Brak danych nie jest zastępowany zerową oceną. "
        "Wyniki są wskaźnikami projektu, a metoda nie została "
        "niezależnie zweryfikowana. Materiał i ślady wody "
        "częściowo opisują te same osady."
    )

    st.write(
        "Hanksville — interpretacja śladów wody: "
        "starsze osady aluwialne oraz jednostka Morrison "
        "obejmująca człony Brushy Basin i Salt Wash. "
        "Interpretacja dotyczy środowisk powstawania "
        "jednostek geologicznych, nie każdego piksela osobno. "
        "Jednostki mieszane pozostają oddzielnie. "
        "Źródło interpretacji Morrison: Utah Geological Survey, "
        "GeoSights: Bentonite Hills."
    )
