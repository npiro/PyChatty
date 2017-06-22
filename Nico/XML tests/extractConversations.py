
from bs4 import BeautifulSoup  # xml parsing library
import codecs  # to open utf-8 file
import sys

filename = '/Users/piromast/Dropbox/S2DS - M&S/Data/full_output_cleaned.xml'


NONE, AGENT, CLIENT, EXTERNAL = 0, 1, 2, 3 # State machine state flags

messageList = []
write_to_file = True

UserCorpus = codecs.open('UserCorpus.txt','a','utf-8')
AgentCorpus = codecs.open('AgentCorpus.txt','a','utf-8')
ExternalTable = codecs.open('ExternalTable.txt','a','utf-8')

clients = 0
agents = 0
externals = 0
for l in codecs.open(filename,'rU', 'utf-8'):
    soup = BeautifulSoup(l, 'xml')
    chats = soup.find_all('chatTranscript')  # Chat root (contains list of chats)
    have_external = False
    for chat in chats:
        parties = chat.find_all("newParty")
        userType, userID = [], []
        
        # Determine clientID and agentID
        for p in parties:
            userType = p.userInfo["userType"]
            userID = p["userId"]
            if userType == "CLIENT":
                clientID = userID
                clients += 1
            elif userType == "AGENT":
                agentID = userID
                agents += 1
            elif userType == 'EXTERNAL':
                externals += 1
                externalID = userID
            else:
                print(parties)
                print(chat.find_all("message"))
                wait = input("PRESS ENTER TO CONTINUE.")
                continue
                print('User type not found')
                sys.exit(1)
            
        messages = chat.find_all("message")
        lastUser = NONE  # 0: NONE, 1: AGENT, 2: CLIENT
        
        
        currentItem = ['','']
        for m in messages:
            text = m.msgText.contents[0]
            userID = m['userId']
            
            if userID == agentID:
                user = AGENT
            elif userID == clientID:
                user = CLIENT
            elif have_external and userID == externalID:
                user = EXTERNAL
                have_external = True
            else:
                continue
            if lastUser == NONE:
                if user == AGENT:
                    lastUser = AGENT
                elif user == CLIENT:
                    lastUser = CLIENT
                    currentItem[0] += text
                else:
                    print('Error: user undefined')
                    break
                
            elif lastUser == AGENT or lastUser == EXTERNAL:
                if user == AGENT or user == EXTERNAL:
                    currentItem[1] += ' ' + text
                    #lastUser = AGENT
                    
                elif user == CLIENT:
                    if len(currentItem[0]) > 0:   
                        if write_to_file:
                            UserCorpus.write(currentItem[0]+'\n')
                            AgentCorpus.write(currentItem[1]+'\n')
                            ExternalTable.write(str(int(have_external)))
                        else:
                            messageList.append(currentItem)
                    currentItem = ['','']
                    currentItem[0] += text
                    lastUser = CLIENT
                    
            elif lastUser == CLIENT:
                if user == AGENT:
                    currentItem[1] += text
                    lastUser = AGENT
                elif user == CLIENT:
                    currentItem[0] += text
            else:
                print('Error: Unknown user type')
                    
        if lastUser == AGENT or lastUser == EXTERNAL:
            if write_to_file:
                UserCorpus.write(currentItem[0])
                AgentCorpus.write(currentItem[1])
            else:
                messageList.append(currentItem)
            