module.exports = function(eleventyConfig){
    eleventyConfig.addPassthroughCopy("src/assets/");
    eleventyConfig.addPlugin(require("eleventy-plugin-heroicons"));

    eleventyConfig.addShortcode(
      "experience",
      (year, place, role, description) =>
      `<div>
      <div class="flex justify-between cursor-pointer" onclick="toggleExpand(this)">
          <div class="flex justify-between w-4/12">
              <p>${year}</p>
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1" stroke="currentColor" class="w-6 h-6">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 0 1 0 1.972l-11.54 6.347a1.125 1.125 0 0 1-1.667-.986V5.653Z" />
                </svg>
                
          </div>
          <div class="w-8/12 flex justify-between items-center">
              <div>${place}</div>
              <div>${role}</div>
          </div>
      </div>
      <div class="hidden flex justify-end">
          <div class="w-8/12">
              <div>${description}</div>
          </div>
      </div>
  </div>`
    )

    return {
      dir: {
        output: "docs",
        input: "src",
        data: "_data",
        includes: "_includes",
        layouts: "_layouts"
      }
    };
  }