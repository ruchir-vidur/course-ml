# Course transcript data

Place WebVTT (`.vtt`) transcript files here, in one directory per course. These files are ignored by Git because they may contain proprietary course content.

After adding or updating transcripts, rebuild the local index from `src`:

```powershell
Set-Location src
python embedding.py
```
