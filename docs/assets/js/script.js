function toggleExpand(element) {
    const content = element.nextElementSibling;
    content.classList.toggle('hidden');
    const svg = element.querySelector('svg');
    svg.style.transition = 'transform 0.3s ease';
    svg.style.transform = svg.style.transform === 'rotate(90deg)' ? 'rotate(0deg)' : 'rotate(90deg)';
}