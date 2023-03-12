from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
fromaddr = "ForestfireAlerts@gmail.com"
toaddr = " "
for i in range(len(toaddr)):
    html = open("")
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr[i]
    msg['Subject'] = "Fire Alerts Report"
    part2 = MIMEText(html.read(), 'html')
    msg.attach(part2)
    debug = False
    if debug:
        print(msg.as_string())
    else:
        server = smtplib.SMTP('smtp.gmail.com',587)
        server.starttls()
        server.login("forestfire.alerts@gmail.com", "**********")
        text = msg.as_string()
        server.sendmail(fromaddr, toaddr[i], text)
        server.quit()