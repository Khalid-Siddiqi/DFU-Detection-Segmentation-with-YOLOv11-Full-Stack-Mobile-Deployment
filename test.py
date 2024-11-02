import base64

# Replace this with the actual base64 string you received
img_base64 = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/..."  # Your base64 string

# Remove the "data:image/jpeg;base64," part
img_data = img_base64.split(",")[1]

# Decode the base64 string
image_data = base64.b64decode(img_data)

# Write the binary data to a file
with open("output_image.jpg", "wb") as f:
    f.write(image_data)

print("Image saved as output_image.jpg")