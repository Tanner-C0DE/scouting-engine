# Scouting Engine
A professional-grade football recruitment tool built with Python and Streamlit. This engine moves beyond standard similarity models by using Priority-Ranked Magnitude Searching, allowing scouts to find players who match both the style and the output volume of world-class benchmarks.

Key Features
--------------------------------------------------------------------
Priority-Ranked Search Engine: Unlike standard models, this engine applies a 3.0x anchor weight to the first statistical factor you select, ensuring your primary scouting requirement (e.g., Goals per 90) drives the results.

Magnitude-Sensitive Matching: Uses Euclidean Distance instead of Cosine Similarity to ensure candidates match the physical volume of a target's stats, preventing "low-output clones" from appearing in results.

Dynamic Radar Overlays: Interactive statistical profiles with custom hover tooltips and smart layering for clear player-to-player benchmarking

Automated Data Pipeline: Cleans and merges player market values, competition data, and advanced performance metrics (Goals/Assists per 90).

Technical Stack
--------------------------------------------------------------------
-Frontend: Streamlit (Custom CSS injected for Midnight/Slate theme).

-Data Processing: Pandas & NumPy.

-Similarity Logic: Scikit-Learn (StandardScaler & Euclidean Vector distance).

-Visualization: Plotly Graph Objects (Scatterpolar).


Install Dependencies
--------------------------------------------------------------------
streamlit

pandas

numpy

plotly

scikit-learn

Data Requirements 
--------------------------------------------------------------------
Ensure your data/ folder contains the following CSV files:
players.csv, clubs.csv, competitions.csv, appearances.csv, and player_valuations.csv.

Launch the Engine
--------------------------------------------------------------------
streamlit run app.py

How to Use
--------------------------------------------------------------------
-Select Benchmark: Choose a target player from the database to use as your scouting "anchor".

-Build Tactical Profile: Select metrics in order of importance. (Note: The first metric selected receives the highest weighting.)

-Set Constraints: Adjust the budget slider and age range to fit your recruitment strategy.

-Analyze: Review the similarity percentages and use the Radar Overlay to visualize how a candidate's shape compares to the target.


*Developed as a proof-of-concept for advanced football analytics.*
