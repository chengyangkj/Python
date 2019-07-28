import winreg
def get_desktop():
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                         r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders')  # ÀûÓÃÏµÍ³µÄÁŽ±í
    return winreg.QueryValueEx(key, "Desktop")[0]  # ·µ»ØµÄÊÇUnicodeÀàÐÍÊýŸÝ
if __name__ == '__main__':
    Desktop_path = str(get_desktop())  # Unicode×ª»¯Îªstr
for name in range(1,11):
    desktop_path=Desktop_path+"\\"
    full_path=desktop_path+str(name)+".txt"
    file=open(full_path,"w")
    file.close()