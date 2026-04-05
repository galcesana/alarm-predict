---
trigger: always_on
---

I am on a Windows machine. 
Always use 'cmd /c' before any shell command you execute in the terminal (e.g., 'cmd /c npm install'). 
This ensures the process terminates correctly and sends an EOF signal. 
Never use naked commands or interactive shells.