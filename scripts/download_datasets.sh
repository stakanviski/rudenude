#!/bin/sh

download() {
    fileid=$1
    filename=$2
    curl -L -o $filename 'https://docs.google.com/uc?export=download&id='$fileid
    unzip $filename
    rm $filename
}

mkdir -p datasets
cd datasets

download 1tQcbu-ZNDReVzHW_BvQbpgrpPD5XokEh girls_naked_standing_edge.zip
download 1I7obEBmt0ga35hsgWv-ybyjtYbgWgjOQ girls_naked_standing.zip
download 1hHt4R9a0Cs_qZGrYnx2DpapqJMOYqr_M girls_dressed_standing.zip
download 1u3WDDFF8-bCE9ON4lN6ShvZEDVSFgovM anime_girls_naked_standing.zip
download 1ZjFL6x8Ugff6P_q4cBquK8CpPcvReF90 girls_naked_standing_edge_filtered.zip
download 1CxwqeOXD88GIkK0UTaqnJm4TRf7i0AT3 girls_nakes_draw.zip
