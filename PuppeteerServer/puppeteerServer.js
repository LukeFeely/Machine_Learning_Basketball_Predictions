"use es6";
/*
This server is essentially a proxy server.
intended to render the html of webpages that use Javascript to render their content.
Using a normal http get, the webpage would be empty only this server is 
able to render the javascript for the page and return the result to the client
*/
const express = require("express");
const puppeteer = require("puppeteer");

const app = express();

/**
 The caller specifies a url that they would like to target using a http query e.g
 http://localhost:4000?url="http://someUrlThatYouWouldLiktToRequest"
 * **/
app.get("/", async (req, res) => {
	const { url } = req.query;
	// extract the url from the query
	if (!url) {
		res.status(400).send("You forgot to include a queryParam");
		return;
	}
	// use puppeteer to load the content in a browser context and return the result to the client
	try {
		const window = await puppeteer.launch();
		const page = await window.newPage();
		await page.goto(pageUrl);
		const pageHTML = await page.evaluate(
			"new XMLSerializer().serializeToString(document.doctype) + document.documentElement.outerHTML"
		);
		await browser.close();
		res.status(200).send(pageHTML);
	} catch (error) {
		res.status(500).send(error);
	}
});

app.listen(port, () =>
	console.log("Puppeteer server listening on port 3000...")
);
