create_fresh_environment:
	conda create --name not_so_close_encounters python=3.5

create_environment:
	conda env create -f environment.yaml

export_environment:
	conda env export > environment.yaml

destroy_environment:
	conda remove --name not_so_close_encounters --all

data/external/cb_2016_us_state_500k.shp:
	wget http://www2.census.gov/geo/tiger/GENZ2016/shp/cb_2016_us_state_500k.zip
	unzip cb_2016_us_state_500k.zip
	mv cb_2016_us_state_500k.* data/external

data/external/military_bases.kml:
	wget https://militarybases.com/map-kml/www.kml
	mv www.kml data/external/military_bases.kml

data/external/military_bases.csv: data/external/military_bases.kml
	python extract_military_bases.py \
		data/external/military_bases.kml \
		data/external/military_bases.csv

plots: data/external/military_bases.csv data/external/cb_2016_us_state_500k.shp
	python build_plots.py