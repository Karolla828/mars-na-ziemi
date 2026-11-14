from pathlib import Path
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


def read_mars_temperature():
    """Profil godzinowy Marsa po podstawowej kontroli jakości."""

    frame = pd.read_csv(
        DATA / "mars_temperatura_sol30.csv"
    )

    temperature = pd.to_numeric(
        frame["GROUND_TEMP"],
        errors="coerce",
    )

    uncertainty = pd.to_numeric(
        frame["GROUND_TEMP_UNCERTAINTY"],
        errors="coerce",
    )

    valid = (
        temperature.between(100, 400)
        & uncertainty.ge(0)
        & np.isfinite(uncertainty)
    )

    # Używamy pomiarów nieruchomego łazika,
    # o małym nachyleniu i bez cienia w polu pomiaru.
    for flag in [
        "ROVER_STILL",
        "ROVER_LOW_TILT",
        "TIRS_GROUND_FOOTPRINT_NOT_IN_SHADOW",
    ]:
        valid &= (
            pd.to_numeric(frame[flag], errors="coerce") == 1
        )

    checked = frame.loc[valid].copy()
    checked["temperature_c"] = temperature.loc[valid] - 273.15

    checked["hour"] = pd.to_numeric(
        checked["LMST"].astype(str).str.extract(
            r"M(\d{2}):",
            expand=False,
        ),
        errors="coerce",
    )

    profile = (
        checked.groupby("hour")["temperature_c"]
        .median()
        .reindex(range(24))
    )

    if profile.isna().any():
        raise ValueError(
            "Po kontroli jakości Mars nie ma pełnego "
            "profilu 24 godzin."
        )

    return profile.to_numpy(dtype=float)


def read_earth_temperature_profiles():
    cells = pd.read_csv(
        DATA / "ziemia_temperatura_region.csv"
    )

    profiles = []

    for _, cell in cells.iterrows():
        latitude = float(cell["latitude"])
        longitude = float(cell["longitude"])

        filename = (
            f"power_ts_{latitude}_{longitude}_202103.json"
        )

        document = json.loads(
            (DATA / filename).read_text(encoding="utf-8")
        )

        values = document["properties"]["parameter"]["TS"]
        fill = document["header"]["fill_value"]

        frame = pd.DataFrame(
            {
                "time": list(values.keys()),
                "temperature_c": list(values.values()),
            }
        )

        frame["temperature_c"] = pd.to_numeric(
            frame["temperature_c"],
            errors="coerce",
        )

        frame = frame[
            np.isfinite(frame["temperature_c"])
            & (frame["temperature_c"] != fill)
        ].copy()

        frame["hour"] = pd.to_numeric(
            frame["time"].str[-2:],
            errors="coerce",
        )

        profile = (
            frame.groupby("hour")["temperature_c"]
            .median()
            .reindex(range(24))
        )

        if profile.isna().any():
            raise ValueError(
                f"Niepełny profil temperatury: {filename}"
            )

        profiles.append(profile.to_numpy(dtype=float))

    if cells.empty:
        raise ValueError("Brak komórek temperatury Ziemi.")

    return cells, np.array(profiles)



def attach_material(earth):
    mars_material = json.loads(
        (DATA / "mars_material.json").read_text(encoding="utf-8")
    )

    material = pd.read_csv(DATA / "ziemia_material.csv")

    classes = [
        "osady_okruchowe",
        "skaly_magmaticzne_i_piroklastyczne",
        "skaly_weglanowe",
        "ewaporaty",
        "skaly_metamorficzne",
    ]

    mars_shares = np.array(
        [mars_material[name] for name in classes],
        dtype=float,
    )

    mars_known = float(mars_shares.sum())

    if not np.isfinite(mars_shares).all() or mars_known <= 0:
        raise ValueError("Brak rozpoznanego materiału Marsa.")

    earth_shares = material[classes].to_numpy(dtype=float)

    if not np.isfinite(earth_shares).all():
        raise ValueError("Nieprawidłowe udziały materiałów Ziemi.")

    earth_known = earth_shares.sum(axis=1)
    valid = earth_known > 0

    # Braku danych nie zamieniamy na zerowe podobieństwo.
    material = material.loc[valid].copy()
    earth_shares = earth_shares[valid]
    earth_known = earth_known[valid]

    if material.empty:
        raise ValueError("Brak rozpoznanego materiału Ziemi.")

    # Porównujemy proporcje WYŁĄCZNIE rozpoznanej części.
    mars_profile = mars_shares / mars_known
    earth_profiles = earth_shares / earth_known[:, None]

    # Wspólna część rozkładów klas: od 0 do 100.
    material["material_score"] = (
        np.minimum(earth_profiles, mars_profile).sum(axis=1) * 100
    )

    material["earth_material_coverage"] = earth_known * 100
    material["mars_material_coverage"] = mars_known * 100

    return earth.merge(
        material[
            [
                "id",
                "material_score",
                "earth_material_coverage",
                "mars_material_coverage",
            ]
        ],
        on="id",
        how="inner",
        validate="one_to_one",
    )



def attach_water(earth):
    reference = json.loads(
        (DATA / "mars_slady_wody.json").read_text(
            encoding="utf-8"
        )
    )

    water = pd.read_csv(DATA / "ziemia_slady_wody.csv")

    if reference["map_coverage"] < 0.95:
        raise ValueError(
            "Mapa procesów wodnych pokrywa za małą część Marsa."
        )

    required = [
        "water_deposits_fraction",
        "mixed_fraction",
        "map_coverage",
    ]

    water = water.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(subset=required).copy()

    # Braków mapy nie uznajemy za brak osadów.
    water = water[water["map_coverage"] >= 0.95].copy()

    if water.empty:
        raise ValueError(
            "Brak obszarów z co najmniej 95% pokrycia mapą."
        )

    # Udziały odnosimy do powierzchni pokrytej mapą.
    mars_fraction = (
        reference["water_deposits_fraction"]
        / reference["map_coverage"]
    )

    earth_fraction = (
        water["water_deposits_fraction"]
        / water["map_coverage"]
    )

    difference = np.abs(earth_fraction - mars_fraction)

    # Umowna skala demonstratora:
    # różnica 20 punktów procentowych daje 50/100.
    water["water_score"] = 100 / (1 + difference / 0.20)

    water["earth_water_percent"] = earth_fraction * 100
    water["mars_water_percent"] = mars_fraction * 100

    water["earth_water_mixed_percent"] = (
        water["mixed_fraction"]
        / water["map_coverage"]
        * 100
    )

    water["mars_water_mixed_percent"] = (
        reference["mixed_fraction"]
        / reference["map_coverage"]
        * 100
    )

    water["water_map_coverage_percent"] = (
        water["map_coverage"] * 100
    )

    return earth.merge(
        water[
            [
                "id",
                "water_score",
                "earth_water_percent",
                "mars_water_percent",
                "earth_water_mixed_percent",
                "mars_water_mixed_percent",
                "water_map_coverage_percent",
            ]
        ],
        on="id",
        how="inner",
        validate="one_to_one",
    )


def calculate_results(use_terrain, use_temperature, weights, use_material=False, use_water=False):
    mars = json.loads(
        (DATA / "mars_cechy.json").read_text(encoding="utf-8")
    )

    earth = pd.read_csv(DATA / "ziemia_cechy.csv")

    if use_water:
        earth = attach_water(earth)

    if use_material:
        earth = attach_material(earth)

    columns = [
        "slope_mean_deg",
        "relief_p95_p05_m",
        "roughness_plane_m",
    ]

    required = columns + ["latitude", "longitude"]

    earth = earth.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(subset=required).copy()

    if earth.empty:
        raise ValueError("Brak poprawnych obszarów Ziemi.")

    mars_values = np.array(
        [mars[column] for column in columns],
        dtype=float,
    )

    if not np.isfinite(mars_values).all():
        raise ValueError("Nieprawidłowe cechy Marsa.")

    earth_values = earth[columns].to_numpy(dtype=float)

    scales = (
        np.percentile(earth_values, 75, axis=0)
        - np.percentile(earth_values, 25, axis=0)
    )

    scales = np.where(
        scales > 1e-9,
        scales,
        np.maximum(np.abs(mars_values), 1.0),
    )

    differences = np.abs(
        earth_values - mars_values
    ) / scales

    terrain_distance = differences @ np.array([0.4, 0.3, 0.3])
    earth["terrain_score"] = 100 / (1 + terrain_distance)

    combined = np.zeros(len(earth), dtype=float)
    total_weight = 0

    if use_terrain:
        combined += earth["terrain_score"].to_numpy() * weights[0]
        total_weight += weights[0]

    thermal_reference = None

    if use_temperature:
        mars_profile = read_mars_temperature()
        cells, earth_profiles = read_earth_temperature_profiles()

        # Porównujemy 24 mediany godzinowe.
        # Każda godzina otrzymuje takie samo znaczenie.
        errors = np.mean(
            np.abs(earth_profiles - mars_profile),
            axis=1,
        )

        # Umowna skala demonstratora:
        # średnia różnica 20°C daje wynik 50/100.
        cell_scores = 100 / (1 + errors / 20.0)

        # Przypisujemy najbliższy środek komórki siatki.
        # To nie tworzy nowych, szczegółowych pomiarów.
        earth_lat = earth["latitude"].to_numpy()
        earth_lon = earth["longitude"].to_numpy()

        cell_lat = cells["latitude"].to_numpy(dtype=float)
        cell_lon = cells["longitude"].to_numpy(dtype=float)

        distances = (
            (earth_lat[:, None] - cell_lat[None, :]) ** 2
            + (
                (earth_lon[:, None] - cell_lon[None, :])
                * np.cos(np.radians(earth_lat[:, None]))
            ) ** 2
        )

        indices = distances.argmin(axis=1)

        earth["temperature_score"] = cell_scores[indices]
        earth["temperature_error_c"] = errors[indices]

        earth["temperature_mean_c"] = (
            earth_profiles.mean(axis=1)[indices]
        )

        earth["temperature_range_c"] = (
            np.ptp(earth_profiles, axis=1)[indices]
        )

        combined += (
            earth["temperature_score"].to_numpy() * weights[1]
        )

        total_weight += weights[1]

        thermal_reference = {
            "mean_c": float(mars_profile.mean()),
            "range_c": float(np.ptp(mars_profile)),
        }

    if use_material:
        combined += earth["material_score"].to_numpy() * weights[2]
        total_weight += weights[2]

    if use_water:
        combined += earth["water_score"].to_numpy() * weights[3]
        total_weight += weights[3]

    earth["score"] = combined / total_weight

    ranking = earth.sort_values(
        "score",
        ascending=False,
    )

    return mars, thermal_reference, ranking


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

    for _, row in ranking.iterrows():
        candidate = row.to_dict()

        if all(
            distance_km(candidate, previous) >= 12
            for previous in selected
        ):
            selected.append(candidate)

        if len(selected) == 10:
            break

    return selected


st.title("🪐 Mars na Ziemi")
st.write(
    "Znajdź ziemskie obszary podobne "
    "do wybranego fragmentu Marsa."
)

st.header("1. Wybierz obszar Marsa")

st.selectbox(
    "Obszar odniesienia",
    ["Jezero — zachodnia delta i przyległe dno"],
)

st.caption(
    "Region poszukiwań: fragment Atakamy. "
    "Porównywane obszary mają około 11,3 × 11,9 km."
)

st.header("2. Wybierz kryteria podobieństwa")

with st.form("criteria_form"):
    left, right = st.columns([1, 2])

    with left:
        terrain_checked = st.checkbox(
            "Rzeźba terenu",
            value=True,
        )

    with right:
        terrain_weight = st.slider(
            "Znaczenie rzeźby terenu",
            0, 100, 70, step=5,
        )

    left, right = st.columns([1, 2])

    with left:
        temperature_checked = st.checkbox(
            "Warunki termiczne — eksperymentalne",
            value=False,
        )

    with right:
        temperature_weight = st.slider(
            "Znaczenie warunków termicznych",
            0, 100, 30, step=5,
        )

    left, right = st.columns([1, 2])

    with left:
        material_checked = st.checkbox(
            "Materiał powierzchni — rozpoznana część",
            value=False,
        )

    with right:
        material_weight = st.slider(
            "Znaczenie materiału powierzchni",
            0, 100, 30, step=5,
        )

    left, right = st.columns([1, 2])

    with left:
        water_checked = st.checkbox(
            "Ślady działania wody — osady rzeczne i jeziorne",
            value=False,
        )

    with right:
        water_weight = st.slider(
            "Znaczenie śladów działania wody",
            0, 100, 30, step=5,
        )

    submitted = st.form_submit_button(
        "Znajdź podobne miejsca",
        type="primary",
    )

st.caption(
    "Odznaczona kategoria lub waga 0 oznacza pominięcie. "
    "Wagi mają znaczenie względem siebie. "
    "Po zmianie ustawień kliknij przycisk wyszukiwania."
)

if submitted:
    st.session_state.pop("results_v4", None)

    use_terrain = terrain_checked and terrain_weight > 0
    use_temperature = (
        temperature_checked and temperature_weight > 0
    )

    use_material = material_checked and material_weight > 0

    use_water = water_checked and water_weight > 0

    if not any([
        use_terrain,
        use_temperature,
        use_material,
        use_water,
    ]):
        st.warning(
            "Wybierz przynajmniej jedną dostępną kategorię "
            "z wagą większą od zera."
        )
    else:
        try:
            with st.spinner("Obliczam dopasowania..."):
                mars, thermal, ranking = calculate_results(
                    use_terrain,
                    use_temperature,
                    [
                        terrain_weight,
                        temperature_weight,
                        material_weight,
                        water_weight,
                    ],
                    use_material=use_material,
                    use_water=use_water,
                )

                st.session_state["results_v4"] = {
                    "mars": mars,
                    "thermal": thermal,
                    "places": select_places(ranking),
                    "use_terrain": use_terrain,
                    "use_temperature": use_temperature,
                    "use_material": use_material,
                    "use_water": use_water,
                }

        except Exception as error:
            st.error(f"Nie udało się obliczyć wyników: {error}")

st.header("3. Mapa podobnych miejsc")

map_view = folium.Map(
    location=[-22.5, -68.5],
    zoom_start=8,
    tiles="OpenStreetMap",
    control_scale=True,
)

results = st.session_state.get("results_v4")

if results:
    for position, place in enumerate(results["places"], start=1):

        lines = []
        notes = []

        if results["use_terrain"]:
            lines.append(
                f"Rzeźba terenu: "
                f"<b>{place['terrain_score']:.1f}/100</b>"
            )

        if results["use_temperature"]:
            lines.append(
                f"Warunki termiczne*: "
                f"<b>{place['temperature_score']:.1f}/100</b>"
            )
            notes.append("* Temperatura: porównanie eksperymentalne.")

        if results.get("use_material", False):
            lines.append(
                f"Materiał powierzchni: "
                f"<b>{place['material_score']:.1f}/100</b>"
            )
            notes.append(
                "Materiał: rozpoznano "
                f"{place['mars_material_coverage']:.0f}% Marsa i "
                f"{place['earth_material_coverage']:.0f}% "
                "obszaru Ziemi."
            )

        if results.get("use_water", False):
            lines.append(
                f"Ślady działania wody: "
                f"<b>{place['water_score']:.1f}/100</b>"
            )
            notes.append(
                "Ślady wody: podobieństwo udziału "
                "zmapowanych osadów."
            )

        categories_html = "<br>".join(lines)

        notes.append("Indeks nie jest procentem zgodności planet.")
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
            <b>Atakama — obszar {int(place['id'])}</b>

            <div style="margin:7px 0 10px;">
                Dopasowanie do Jezero:
                <b>{place['score']:.1f}/100</b>
            </div>

            {categories_html}

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
            [place["latitude"], place["longitude"]],
            tooltip=folium.Tooltip(content, sticky=False),
            popup=folium.Popup(content, max_width=350),
            icon=folium.Icon(
                color="red" if position == 1 else "blue",
                icon="info-sign",
            ),
        ).add_to(map_view)

    st.caption(
        "Najedź na pinezkę lub ją kliknij. "
        "Czerwona oznacza najlepszy pokazany wynik. "
        "Środki miejsc dzieli co najmniej 12 km. "
        "Przy remisach podobnie dobre obszary mogą nie być pokazane."
    )

    if results["use_temperature"]:
        st.caption(
            "Warunki termiczne są kategorią eksperymentalną. "
            "Porównujemy temperaturę jasnościową gruntu Marsa "
            "z temperaturą powierzchni z reanalizy Ziemi. "
            "To nie jest pełne porównanie klimatu."
        )
else:
    st.caption(
        "Wybierz kryteria i kliknij „Znajdź podobne miejsca”."
    )

st_folium(
    map_view,
    height=570,
    use_container_width=True,
    returned_objects=[],
    key="map_v2",
)

with st.expander("Źródła danych i sposób liczenia"):
    st.write(
        "Ślady działania wody: mapa Jezero USGS Sun i Stack (2020) "
        "oraz zbiór Geología Base z portalu Ministerstwa "
        "Środowiska Chile, w skali 1:1 000 000. "
        "Porównujemy udział jednoznacznie opisanych osadów "
        "rzecznych, aluwialnych lub jeziornych w powierzchni "
        "pokrytej mapą. Jednostki mieszane pozostają osobno. "
        "Ziemskie obszary z pokryciem poniżej 95% są pomijane."
    )

    st.write(
        "Indeks procesów wodnych = 100 / "
        "(1 + bezwzględna różnica udziałów / 0,20). "
        "Różnica 20 punktów procentowych daje 50/100. "
        "To umowna skala, nie zweryfikowana norma naukowa. "
        "Mapy obu planet mają różną szczegółowość. "
        "Materiał i procesy wodne częściowo opisują te same "
        "osady, więc kategorie nie są niezależne."
    )
    st.write(
        "Materiał powierzchni: mapa geologiczna USGS "
        "Sun i Stack (2020) dla Jezero oraz mapa GLiM dla Ziemi. "
        "Porównujemy udziały szerokich klas w rozpoznanej części. "
        "Niepewne jednostki pozostają nieokreślone. "
        "Wynik nie potwierdza zgodności całego obszaru."
    )
    st.write(
        "Rzeźba: CTX/USGS/NASA MRO dla Marsa, "
        "Terrain Tiles dla Ziemi; analiza na siatce 100 m."
    )

    st.write(
        "Temperatura: MEDA/TIRS, sol 30, dla Marsa; "
        "NASA POWER/MERRA-2, marzec 2021, dla Ziemi. "
        "Porównujemy 24 mediany godzinowe według czasu lokalnego. "
        "Nieznana flaga położenia koła łazika pozostaje "
        "ograniczeniem jakości danych Marsa."
    )

    st.write(
        "Wynik termiczny = 100 / (1 + średnia bezwzględna "
        "różnica profili / 20°C). Skala 20°C jest umownym "
        "parametrem demonstratora, a nie normą naukową."
    )

    st.write(
        "Wynik rzeźby wykorzystuje nachylenie, przewyższenia "
        "i nierówność z wagami 40/30/30. Różnice skalujemy "
        "rozstępem międzykwartylowym cech w regionie Ziemi."
    )

    st.write(
        "Wynik łączny jest średnią wyników aktywnych kategorii, "
        "ważoną ustawieniami suwaków. Metoda nie została "
        "jeszcze niezależnie zweryfikowana."
    )