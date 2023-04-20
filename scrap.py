from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import pandas as pd


def ScrapComment(url):
    option = webdriver.ChromeOptions()     #use to provide chrome drive option like setting path,adding arguments, extensions.etc
    option.add_argument("--headless")      #to run browser in background without opeing the window
    driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(), options=option)
    driver.get(url)     #access link
    prev_h = 0   #set web page height parameter to zero

###loop over the webpage to scroll the comments till the last comment

    while True:
        #js script code to get the height of web page
        height = driver.execute_script("""
                function getActualHeight() {
                    return Math.max(
                        Math.max(document.body.scrollHeight, document.documentElement.scrollHeight),
                        Math.max(document.body.offsetHeight, document.documentElement.offsetHeight),
                        Math.max(document.body.clientHeight, document.documentElement.clientHeight)
                    );
                }
                return getActualHeight();
            """)
        driver.execute_script(f"window.scrollTo({prev_h},{prev_h + 200})") #scrolling the page comments
        # fix the time sleep value according to your network connection
        time.sleep(1)
        prev_h +=200    
        if prev_h >= height:
            break
    soup = BeautifulSoup(driver.page_source, 'html.parser')   # to parse info from html 
    driver.quit()
    title_text_div = soup.select_one('#container h1') 
    title = title_text_div and title_text_div.text   
    comment_div = soup.select("#content #content-text")
    comment_list = [x.text for x in comment_div]
    #print(title, comment_list)
    dict1={'Comment':comment_list}
    comments_df= pd.DataFrame(dict1)
    #print(df)
    return comments_df


if __name__ == "__main__":
    ScrapComment()
    