import pdfplumber
from PIL import Image
import os
import subprocess
from data_storage import scp_transfer  # Import the SCP transfer function


# Input PDF file paths
pdf1_path = "comparison_max_eps_metallicity_rms_vz_roche.pdf"
pdf2_path = "mass_fraction_raettig.pdf"
output_path = "merged_raettig.pdf"

# Open the first PDF and convert the first page to an image
with pdfplumber.open(pdf1_path) as pdf1:
    page1 = pdf1.pages[0]
    image1 = page1.to_image(resolution=300)
    image1_path = "image1.png"
    image1.save(image1_path)

# Open the second PDF and convert the first page to an image
with pdfplumber.open(pdf2_path) as pdf2:
    page2 = pdf2.pages[0]
    image2 = page2.to_image(resolution=300)
    image2_path = "image2.png"
    image2.save(image2_path)

# Load the images using PIL
image1 = Image.open(image1_path)
image2 = Image.open(image2_path)

# Calculate the total height for the new image
total_width = max(image1.width, image2.width)
total_height = image1.height + image2.height

# Create a new blank image with the total height
new_image = Image.new("RGB", (total_width, total_height), "white")
new_image.paste(image1, (0, 0))
new_image.paste(image2, (0, image1.height))

# Save the combined image as a PDF
new_image.save(output_path, "PDF", resolution=300)
print(f"Merged PDF saved as {output_path}")

# Transfer the file to your laptop using scp
scp_transfer(output_path, "/Users/mariuslehmann/Downloads/Profiles", "mariuslehmann")
