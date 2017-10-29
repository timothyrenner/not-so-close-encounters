import click

from lxml import etree
from csv import DictWriter

@click.command()
@click.argument('input_file', type=click.File('r'))
@click.argument('output_file', type=click.File('w'))
def main(input_file, output_file):
    namespaces={"kml":"http://www.opengis.net/kml/2.2"}
    military_base_tree = etree.parse(input_file)
    
    names = military_base_tree.xpath(
        '//kml:Placemark/kml:name/text()',
        namespaces=namespaces
    )

    coordinates = military_base_tree.xpath(
        '//kml:Placemark/kml:Point/kml:coordinates/text()',
        namespaces=namespaces
    )

    # Each of these needs to be parsed and extracted as HTML.
    # This will happen inside the loop.
    description_tables = military_base_tree.xpath(
        '//kml:Placemark/kml:description/text()',
        namespaces=namespaces
    )

    # HTML parser for the tables.
    html_parser = etree.HTMLParser()

    # Dict writer for the output csv.
    fieldnames = [
        "name",
        "branch",
        "link",
        "latitude",
        "longitude"
    ]

    writer = DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()

    for name, coordinate_str, description \
        in zip(names, coordinates, description_tables):
        
        base_lon, base_lat = coordinate_str.split(',')

        description_tree = etree.fromstring(description, html_parser)
        branch = description_tree.xpath(
            '//table/tr/td'
        )[1].xpath( # Second row has the branch name.
            './b/text()'
        )[0]
        link = description_tree.xpath(
            '//table/tr/td'
        )[0].xpath(
            './a/@href'
        )[0]

        writer.writerow(
            {
                "name": name,
                "branch": branch,
                "link": link,
                "latitude": base_lat,
                "longitude": base_lon
            }
        )

if __name__ == "__main__":
    main()