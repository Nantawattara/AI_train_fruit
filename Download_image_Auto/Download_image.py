import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service

def download_images(fruit_name):
    # Create directory with full path
    save_dir = os.path.join("D:\\scr\\AI_train_fruit\\Download_image_Auto", fruit_name)
    os.makedirs(save_dir, exist_ok=True)

    service = Service("C:\\Users\\focus\\Downloads\\edgedriver_win64\\msedgedriver.exe")
    driver = webdriver.Edge(service=service)

    # Navigate to Google Images
    driver.get(f"https://www.google.com/search?q={fruit_name}&tbm=isch")

    # Scroll down the page to load more images
    for _ in range(5):
        driver.execute_script("window.scrollBy(0,1000);")
        time.sleep(2)

    # Fetch image URLs
    image_elements = driver.find_elements(By.CSS_SELECTOR, "img")
    image_urls = [img.get_attribute("src") for img in image_elements if img.get_attribute("src") and "http" in img.get_attribute("src")]

    # Close the webdriver
    driver.quit()

    # Download and save images
    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                # Save with full path to the target directory
                save_path = os.path.join(save_dir, f"{fruit_name}_{i+1}.jpg")
                with open(save_path, "wb") as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                print(f"✅ Successfully downloaded image {i+1} of {fruit_name}")
            else:
                print(f"❌ Image {i+1} of {fruit_name} could not be downloaded (HTTP {response.status_code})")
        except Exception as e:
            print(f"⚠️ Error with image {i+1} of {fruit_name}: {e}")

    print(f"🎉 Finished downloading images of {fruit_name}!")

# Call the function for each fruit
fruits = ["ผลตะขบ", "ผลไม้ลูกมะพร้าว", "ผลไม้ลูกมังคุด", "ผลไม้ลูกลำไย", "ผลไม้ทุเรียน"]
for fruit in fruits:
    download_images(fruit)