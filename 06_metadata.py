import datetime
import os
import sqlite3

import yaml

# Define input and output GeoPackage file paths as constants
INPUT_GPKG = "data/05_stat/gpkg/austria_new.gpkg"  # Replace with your input GeoPackage file
OUTPUT_GPKG = "output.gpkg"  # Replace with your output GeoPackage file
YAML_FILE = "metadata/metadata.yml"


def get_inspire_test_metadata():
    """
    Returns a sample INSPIRE metadata dictionary for testing purposes.
    """
    inspire_metadata = {
        "fileIdentifier": "unique_file_identifier_12345",
        "language": {"codeList": "http://www.loc.gov/standards/iso639-2/php/code_list.php", "codeListValue": "eng"},
        "characterSet": {
            "codeList": "http://standards.iso.org/iso/19139/resources/gmxCodelists.xml#MD_CharacterSetCode",
            "codeListValue": "utf8",
        },
        "hierarchyLevel": {
            "codeList": "http://standards.iso.org/iso/19139/resources/gmxCodelists.xml#MD_ScopeCode",
            "codeListValue": "dataset",
        },
        "contact": {
            "organisationName": "Example Organisation",
            "contactInfo": {"address": {"electronicMailAddress": "info@example.com"}},
            "role": {
                "codeList": "http://standards.iso.org/iso/19139/resources/gmxCodelists.xml#CI_RoleCode",
                "codeListValue": "pointOfContact",
            },
        },
        "dateStamp": "2024-07-24",
        "metadataStandardName": "ISO 19115",
        "metadataStandardVersion": "ISO 19115:2003/Cor.1:2006",
        "identificationInfo": {
            "citation": {
                "title": "Example GeoPackage Dataset",
                "date": {
                    "date": "2024-01-15",
                    "dateType": {
                        "codeList": "http://standards.iso.org/iso/19139/resources/gmxCodelists.xml#CI_DateTypeCode",
                        "codeListValue": "publication",
                    },
                },
                "identifier": {"code": "example_dataset_id_001"},
            },
            "abstract": "This GeoPackage contains example spatial data, including points, lines, and polygons, for demonstration purposes.  It is compliant with the INSPIRE Directive.",
            "purpose": "To demonstrate the structure of a GeoPackage with INSPIRE-compliant metadata.",
            "status": {
                "codeList": "http://standards.iso.org/iso/19139/resources/gmxCodelists.xml#MD_ProgressCode",
                "codeListValue": "completed",
            },
            "pointOfContact": {
                "organisationName": "Example Organisation",
                "contactInfo": {"address": {"electronicMailAddress": "info@example.com"}},
                "role": {
                    "codeList": "http://standards.iso.org/iso/19139/resources/gmxCodelists.xml#CI_RoleCode",
                    "codeListValue": "originator",
                },
            },
            "resourceMaintenance": {
                "maintenanceFrequency": {
                    "codeList": "http://standards.iso.org/iso/19139/resources/gmxCodelists.xml#MD_MaintenanceFrequencyCode",
                    "codeListValue": "asNeeded",
                },
                "maintenanceNote": "Metadata is updated as needed.",
            },
            "descriptiveKeywords": [
                {"keyword": "GeoPackage"},
                {"keyword": "INSPIRE"},
                {"keyword": "Spatial Data"},
                {"keyword": "Point"},
                {"keyword": "Line"},
                {"keyword": "Polygon"},
                {
                    "keyword": "INSPIRE data theme",
                    "thesaurusName": {
                        "title": "INSPIRE Spatial Data Themes",
                        "date": {
                            "date": "2008-06-01",
                            "dateType": {
                                "codeList": "http://standards.iso.org/iso/19139/resources/gmxCodelists.xml#CI_DateTypeCode",
                                "codeListValue": "publication",
                            },
                        },
                    },
                },
            ],
            "resourceConstraints": {
                "useLimitation": "No restrictions apply.",
                "accessConstraints": {
                    "codeList": "http://standards.iso.org/iso/19139/resources/gmxCodelists.xml#MD_RestrictionCode",
                    "codeListValue": "otherRestrictions",
                },
                "otherConstraints": "According to the INSPIRE Directive, access to this resource is allowed under the following conditions [describe conditions].",
            },
            "spatialRepresentationType": {
                "codeList": "http://standards.iso.org/iso/19139/resources/gmxCodelists.xml#MD_SpatialRepresentationTypeCode",
                "codeListValue": "vector",
            },
            "spatialResolution": {"equivalentScale": {"denominator": 10000}},
            "referenceSystemInfo": {"referenceSystemIdentifier": {"code": "EPSG:4326", "codeSpace": "EPSG"}},
            "extent": {
                "geographicElement": {
                    "westBoundLongitude": -10.0,
                    "eastBoundLongitude": 25.0,
                    "southBoundLatitude": 35.0,
                    "northBoundLatitude": 70.0,
                },
                "temporalElement": {"extent": {"beginPosition": "2020-01-01", "endPosition": "2024-12-31"}},
            },
            "resourceFormat": {"name": "GeoPackage", "version": "1.2"},
            "resourceLocator": [
                {
                    "linkage": "http://example.com/geopackage_download.gpkg",
                    "description": "Download the GeoPackage",
                    "function": {
                        "codeList": "http://standards.iso.org/iso/19139/resources/gmxCodelists.xml#CI_OnLineFunctionCode",
                        "codeListValue": "download",
                    },
                }
            ],
            "lineage": {
                "statement": "The data in this GeoPackage was created by [Process Description].  The data was converted to GeoPackage format from [Original Format] on [Date].  Metadata was created following the INSPIRE Metadata Implementing Rules.",
                "processStep": {
                    "description": "Data conversion from source format to GeoPackage.",
                    "dateTime": "2024-01-10T10:00:00",
                    "source": {"description": "Original data source: [Name and format of original data]."},
                },
            },
            "dataQualityInfo": {
                "scope": {
                    "level": {
                        "codeList": "http://standards.iso.org/iso/19139/resources/gmxCodelists.xml#DQ_ScopeCode",
                        "codeListValue": "dataset",
                    }
                },
                "report": {
                    "domainConsistency": {
                        "result": {
                            "conformanceResult": {
                                "specification": {
                                    "title": "Commission Regulation (EU) No 1089/2010 of 23 November 2010 implementing Directive 2007/2/EC of the European Parliament and of the Council as regards interoperability of spatial data sets and services",
                                    "date": {
                                        "date": "2010-12-08",
                                        "dateType": {
                                            "codeList": "http://standards.iso.org/iso/19139/resources/gmxCodelists.xml#CI_DateTypeCode",
                                            "codeListValue": "publication",
                                        },
                                    },
                                    "explanation": "This dataset is conformant with the INSPIRE Implementing Rules for metadata and spatial data sets.",
                                    "pass": True,
                                }
                            }
                        }
                    }
                },
            },
        },
    }
    return inspire_metadata


def write_inspire_metadata_to_yaml(metadata, filepath="inspire_metadata.yaml"):
    """
    Writes an INSPIRE metadata dictionary to a YAML file.

    Args:
        metadata (dict): A dictionary representing INSPIRE metadata.
        filepath (str, optional): The path to the YAML file to write.
            Defaults to "inspire_metadata.yaml".

    Raises:
        TypeError: If metadata is not a dictionary or filepath is not a string.
        yaml.YAMLError: If an error occurs during YAML serialization.
        OSError: If there is an issue writing to the file.
    """
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary.")
    if not isinstance(filepath, str):
        raise TypeError("filepath must be a string.")

    try:
        with open(filepath, "w") as file:
            yaml.dump(metadata, file, indent=2)  # Use indent for better readability
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error writing to YAML file: {e}") from e
    except OSError as e:
        raise OSError(f"Error writing to file: {e}") from e


def read_inspire_metadata_from_yaml(filepath):
    """
    Reads INSPIRE metadata from a YAML file and returns it as a dictionary.

    Args:
        filepath (str): The path to the YAML file.

    Returns:
        dict: A dictionary representing the INSPIRE metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        TypeError: If filepath is not a string.
        yaml.YAMLError: If there is an error parsing the YAML.
    """
    if not isinstance(filepath, str):
        raise TypeError("filepath must be a string.")
    try:
        with open(filepath, "r") as file:
            metadata = yaml.safe_load(file)
        return metadata or {}
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {filepath}") from e
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML: {e}") from e


def remove_gpkg_metadata(input_gpkg_path, output_gpkg_path):
    """
    Removes all metadata from all layers of a GeoPackage file and saves the
    modified GeoPackage to a new file. This function directly manipulates the
    GeoPackage's internal SQLite database.  It deletes entries from the
    'gpkg_metadata' and 'gpkg_metadata_reference' tables. It also clears the
    application_id.

    Args:
        input_gpkg_path (str): The path to the input GeoPackage file.
        output_gpkg_path (str): The path to the output GeoPackage file.

    Returns:
        None: The function modifies the GeoPackage file in place.

    Raises:
        sqlite3.Error: If any error occurs during database operations.
        FileNotFoundError: If the input GeoPackage file does not exist.
        TypeError: If either gpkg_path is not a string.
        OSError: If the output file already exists and cannot be overwritten.
    """
    if not isinstance(input_gpkg_path, str) or not isinstance(output_gpkg_path, str):
        raise TypeError("gpkg_path must be a string.")
    if not os.path.exists(input_gpkg_path):
        raise FileNotFoundError(f"GeoPackage file not found: {input_gpkg_path}")
    if os.path.exists(output_gpkg_path):
        raise OSError(f"Output GeoPackage file already exists: {output_gpkg_path}")

    try:
        # Establish a connection to the input GeoPackage file.
        input_conn = sqlite3.connect(input_gpkg_path)
        input_cursor = input_conn.cursor()

        # Establish a connection to the output GeoPackage file.
        output_conn = sqlite3.connect(output_gpkg_path)
        output_cursor = output_conn.cursor()

        # Copy the entire database structure and data
        input_conn.backup(output_conn)

        # Delete all entries from the gpkg_metadata table in the output.
        output_cursor.execute("DELETE FROM gpkg_metadata")

        # Delete all entries from the gpkg_metadata_reference table in the output.
        output_cursor.execute("DELETE FROM gpkg_metadata_reference")

        # Clear the application_id in the output.
        output_cursor.execute("PRAGMA application_id = 0")

        # Commit the changes to the output database.
        output_conn.commit()

        print(f"Metadata removed from all layers in GeoPackage: {output_gpkg_path}")

    except sqlite3.Error as e:
        # Handle any SQLite errors, rollback changes, and re-raise.
        if output_conn:
            output_conn.rollback()
        raise sqlite3.Error(f"Error accessing GeoPackage: {e}") from e
    finally:
        # Ensure the database connections are closed, even if errors occur.
        if input_conn:
            input_conn.close()
        if output_conn:
            output_conn.close()


def manipulate_inspire_metadata(metadata, gpkg_path, layer_name):
    """
    Manipulates an INSPIRE metadata dictionary to match the properties of a specific GeoPackage layer,
    calculating extent and EPSG from the GeoPackage itself.

    Args:
        metadata (dict): A dictionary representing INSPIRE metadata.
        gpkg_path (str): The path to the GeoPackage file.
        layer_name (str): The name of the layer within the GeoPackage.

    Returns:
        dict: A modified INSPIRE metadata dictionary.

    Raises:
        TypeError: If metadata, gpkg_path or layer_name have wrong type.
        ValueError: If the layer is not found in the GeoPackage.
        sqlite3.Error: If an error occurs while querying the GeoPackage.
    """
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary.")
    if not isinstance(gpkg_path, str):
        raise TypeError("gpkg_path must be a string.")
    if not isinstance(layer_name, str):
        raise TypeError("layer_name must be a string.")

    try:
        conn = sqlite3.connect(gpkg_path)
        cursor = conn.cursor()

        # Get the layer's bounds
        cursor.execute(
            """
            SELECT min_x, min_y, max_x, max_y FROM rtree_{layer_name}
            """.format(layer_name=layer_name)
        )
        bounds = cursor.fetchone()
        if not bounds:
            cursor.execute(
                """
                SELECT min(ST_X(geom)), min(ST_Y(geom)), max(ST_X(geom)), max(ST_Y(geom)) FROM {layer_name}
                """.format(layer_name=layer_name)
            )
            bounds = cursor.fetchone()
        if not bounds:
            raise ValueError(f"Layer '{layer_name}' not found in GeoPackage or has no spatial data.")
        minx, miny, maxx, maxy = bounds

        # Get the layer's SRID (EPSG)
        cursor.execute(
            """
            SELECT srs_id FROM gpkg_contents WHERE table_name = ?
            """,
            (layer_name,),
        )
        srid_result = cursor.fetchone()
        if not srid_result:
            raise ValueError(f"Layer '{layer_name}' not found in GeoPackage or missing SRID.")
        srid = srid_result[0]

        cursor.close()
        conn.close()

    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error accessing GeoPackage: {e}") from e

    modified_metadata = metadata.copy()

    # Update extent
    modified_metadata["identificationInfo"]["extent"]["geographicElement"] = {
        "westBoundLongitude": minx,
        "eastBoundLongitude": maxx,
        "southBoundLatitude": miny,
        "northBoundLatitude": maxy,
    }

    # Update EPSG
    modified_metadata["referenceSystemInfo"]["referenceSystemIdentifier"]["code"] = f"EPSG:{srid}"

    # Update dateStamp and citation date
    current_datetime = datetime.datetime.now()
    modified_metadata["dateStamp"] = current_datetime.strftime("%Y-%m-%d")
    modified_metadata["identificationInfo"]["citation"]["date"]["date"] = current_datetime.strftime("%Y-%m-%d")

    return modified_metadata


def write_metadata_to_gpkg(gpkg_path, layer_name, metadata):
    """
    Writes INSPIRE metadata to a GeoPackage layer.

    Args:
        gpkg_path (str): The path to the GeoPackage file.
        layer_name (str): The name of the layer.
        metadata (dict): The INSPIRE metadata dictionary.

    Raises:
        sqlite3.Error: If any error occurs during database operations.
        TypeError: If any of the arguments has an invalid type.
    """
    if not isinstance(gpkg_path, str):
        raise TypeError("gpkg_path must be a string.")
    if not isinstance(layer_name, str):
        raise TypeError("layer_name must be a string.")
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary.")
    try:
        conn = sqlite3.connect(gpkg_path)
        cursor = conn.cursor()

        # Clear existing metadata for the layer
        cursor.execute(
            "DELETE FROM gpkg_metadata WHERE md_file_id IN (SELECT id FROM gpkg_metadata_reference WHERE table_name = ?)",
            (layer_name,),
        )
        cursor.execute("DELETE FROM gpkg_metadata_reference WHERE table_name = ?", (layer_name,))

        # Convert the metadata dictionary to a string (e.g., using yaml.dump or json.dumps)
        metadata_string = yaml.dump(metadata)

        # Insert the metadata
        cursor.execute("INSERT INTO gpkg_metadata (md_content) VALUES (?)", (metadata_string,))
        md_id = cursor.lastrowid

        # Reference the metadata to the layer
        cursor.execute(
            "INSERT INTO gpkg_metadata_reference (md_file_id, table_name, mime_type) VALUES (?, ?, 'text/xml')",
            (md_id, layer_name),
        )

        conn.commit()
        cursor.close()
        conn.close()
        print(f"Metadata successfully written to layer '{layer_name}' in GeoPackage: {gpkg_path}")

    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error writing metadata to GeoPackage: {e}") from e


if __name__ == "__main__":
    # 1) Remove the metadata for all layers from the GeoPackage
    remove_gpkg_metadata(INPUT_GPKG, OUTPUT_GPKG)

    # 2) Read the metadata from the YAML file
    inspire_metadata = read_inspire_metadata_from_yaml(YAML_FILE)

    # 3) Get the layers from the geopackage
    conn = sqlite3.connect(OUTPUT_GPKG)  # connect to the output file
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'rtree_%'"
    )
    layers = [layer[0] for layer in cursor.fetchall()]
    cursor.close()
    conn.close()

    # 4) Loop through the layers, modify the metadata, and write it to the GeoPackage
    for layer in layers:
        print(f"Processing layer: {layer}")
        modified_metadata = manipulate_inspire_metadata(inspire_metadata, OUTPUT_GPKG, layer)
        write_metadata_to_gpkg(OUTPUT_GPKG, layer, modified_metadata)
